import sys
import logging
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.utils import create_dense_mask_from_landmarks, encode_mask_to_rle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandmarkProcessor:
    """Process landmarks and create segmentation masks for chest X-ray images."""
    
    LANDMARK_SPLITS = {
        'right_lung': slice(0, 44),
        'left_lung': slice(44, 94),
        'heart': slice(94, None)
    }
    

    def __init__(self, annotations_path: Path, padding_path: Path, batch_size: int = 100, max_workers: int = 10):
        """
        Initialize the processor with paths to annotation and padding data.
        
        Args:
            annotations_path: Path to the annotations CSV file
            padding_path: Path to the padding information CSV file
            batch_size: Number of images to process in each batch
            max_workers: Maximum number of worker processes for parallel processing
        """
        # Read CSVs efficiently with specific dtypes
        dtype_dict = {
            'image_id': str,
            'Dice RCA (Mean)': float,
            'Dice RCA (Max)': float,
            'Landmarks': str
        }
        self.annotations_df = pd.read_csv(annotations_path, dtype=dtype_dict)
        
        padding_dtype_dict = {
            'filename': str,
            'height': int,
            'width': int,
            'pad_left': int,
            'pad_top': int
        }
        padding_df = pd.read_csv(padding_path, dtype=padding_dtype_dict)
        self.padding_df = self._preprocess_padding_data(padding_df)
        
        # Create an index for faster lookups
        self.padding_df.set_index('filename', inplace=True)
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._verify_data_integrity()


    def _verify_data_integrity(self):
        """Verify that all image IDs in annotations have corresponding padding information."""
        missing_paddings = set(self.annotations_df['image_id']) - set(self.padding_df.index)
        if missing_paddings:
            logger.warning(f"Missing padding information for {len(missing_paddings)} images")
            logger.warning(f"First few missing IDs: {list(missing_paddings)[:5]}")


    @staticmethod
    def _preprocess_padding_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess padding data."""
        df['filename'] = df['filename'].str.replace('.dicom', '')
        return df


    @staticmethod
    def _process_single_item(item: Tuple[pd.Series, pd.Series]) -> Dict[str, Any]:
        """Process a single image's landmarks and create masks."""
        row, padding_info = item
        
        # Process landmarks
        landmarks = np.array(eval(row["Landmarks"]))
        landmarks = landmarks.reshape(-1, 2) / 1024

        max_shape = max(padding_info["height"], padding_info["width"])
        landmarks = landmarks * max_shape

        # Adjust for padding
        landmarks[:, 0] -= padding_info["pad_left"]
        landmarks[:, 1] -= padding_info["pad_top"]
        processed_landmarks = np.round(landmarks).astype(int)
        landmarks_str = ', '.join(map(str, processed_landmarks.flatten()))

        # Create masks
        image_shape = (int(padding_info["height"]), int(padding_info["width"]))
        masks = {}
        for region, slice_idx in LandmarkProcessor.LANDMARK_SPLITS.items():
            mask = create_dense_mask_from_landmarks(processed_landmarks[slice_idx], image_shape)
            masks[region] = encode_mask_to_rle(mask)
        
        return {
            "image_id": row["image_id"],
            "Dice RCA (Mean)": row["Dice RCA (Mean)"],
            "Dice RCA (Max)": row["Dice RCA (Max)"],
            "Landmarks": landmarks_str,
            "Left Lung": masks['left_lung'],
            "Right Lung": masks['right_lung'],
            "Heart": masks['heart'],
            "Height": padding_info["height"],
            "Width": padding_info["width"]
        }


    def _process_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a batch of images in parallel."""
        # Prepare items for processing
        items = [
            (row, self.padding_df.loc[row["image_id"]])
            for _, row in batch_df.iterrows()
        ]
        
        # Process items in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_single_item, items))
        
        return results


    def process(self, output_path: Path) -> None:
        """Process all images in batches and save results to CSV."""
        processed_data = []
        total_batches = len(self.annotations_df) // self.batch_size + (1 if len(self.annotations_df) % self.batch_size else 0)
        
        for i in tqdm(range(0, len(self.annotations_df), self.batch_size), 
                     total=total_batches, desc="Processing batches"):
            batch_df = self.annotations_df.iloc[i:i + self.batch_size]
            batch_results = self._process_batch(batch_df)
            processed_data.extend(batch_results)
        
        # Create and save the final processed DataFrame
        pd.DataFrame(processed_data).to_csv(output_path, index=False)
        
        # Clean up temporary file
        temp_file = output_path.with_suffix('.temp.csv')
        if temp_file.exists():
            temp_file.unlink()
            
        logger.info(f"Processed data saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process landmarks and create segmentation masks for chest X-ray images.'
    )
    
    parser.add_argument('--annotations-path', type=Path, required=True, help='Path to the annotations CSV file')
    parser.add_argument('--padding-path', type=Path, required=True, help='Path to the padding information CSV file')
    parser.add_argument('--output-path', type=Path, required=True, help='Path where the processed CSV file will be saved')

    parser.add_argument('--batch-size', type=int, default=100, help='Number of images to process in each batch (default: 100)')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of worker processes for parallel processing (default: 10)')

    return parser.parse_args()


def main():
    args = parse_args()
    
    processor = LandmarkProcessor(
        annotations_path=args.annotations_path,
        padding_path=args.padding_path,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    processor.process(output_path=args.output_path)


if __name__ == "__main__":
    main()
