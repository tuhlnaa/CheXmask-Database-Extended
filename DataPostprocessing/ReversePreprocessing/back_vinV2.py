
import sys
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Any

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.utils import create_dense_mask_from_landmarks, encode_mask_to_rle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LandmarkProcessor:
    """Process landmarks and create segmentation masks for chest X-ray images."""
    
    LANDMARK_SPLITS = {
        'right_lung': slice(0, 44),
        'left_lung': slice(44, 94),
        'heart': slice(94, None)
    }
    
    def __init__(self, annotations_path: Path, padding_path: Path):
        """Initialize the processor with paths to annotation and padding data."""
        self.annotations_df = pd.read_csv(annotations_path)
        self.padding_df = self._preprocess_padding_data(pd.read_csv(padding_path))
        self._verify_data_integrity()


    def _verify_data_integrity(self):
        """Verify that all image IDs in annotations have corresponding padding information."""
        print(type(self.annotations_df['image_id']))
        missing_paddings = set(self.annotations_df['image_id']) - set(self.padding_df['filename'])
        if missing_paddings:
            logger.warning(f"Missing padding information for {len(missing_paddings)} images")
            logger.warning(f"First few missing IDs: {list(missing_paddings)[:5]}")


    @staticmethod
    def _preprocess_padding_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess padding data."""
        #df['filename'] = df['filename'].str.replace('.dicom', '').str.split('/').str[-1]
        df['filename'] = df['filename'].str.replace('.dicom', '')
        return df
    

    def _process_landmarks(self, landmarks: np.ndarray, 
                          height: int, width: int, 
                          pad_left: int, pad_top: int) -> np.ndarray:
        """
        Process landmarks by scaling and adjusting for padding.
        
        Args:
            landmarks: Array of landmark coordinates
            height: Image height
            width: Image width
            pad_left: Left padding
            pad_top: Top padding
            
        Returns:
            Processed landmark coordinates
        """
        landmarks = landmarks.reshape(-1, 2) / 1024
        max_shape = max(height, width)
        landmarks = landmarks * max_shape
        
        # Adjust for padding
        landmarks[:, 0] -= pad_left
        landmarks[:, 1] -= pad_top
        
        return np.round(landmarks).astype(int)
    

    def _create_masks(self, landmarks: np.ndarray, 
                      image_shape: Tuple[int, int]) -> Dict[str, str]:
        """
        Create and encode masks for each anatomical region.
        
        Args:
            landmarks: Processed landmark coordinates
            image_shape: Tuple of (height, width)
            
        Returns:
            Dictionary of RLE-encoded masks for each region
        """
        masks = {}
        for region, slice_idx in self.LANDMARK_SPLITS.items():
            mask = create_dense_mask_from_landmarks(landmarks[slice_idx], image_shape)
            masks[region] = encode_mask_to_rle(mask)
        return masks
    

    def _create_row(self, row: pd.Series, 
                    padding_info: pd.Series, 
                    masks: Dict[str, str]) -> Dict[str, Any]:
        """Create a new row with processed data."""
        return {
            "image_id": row["image_id"],
            "Dice RCA (Mean)": row["Dice RCA (Mean)"],
            "Dice RCA (Max)": row["Dice RCA (Max)"],
            "Landmarks": row["Landmarks"],
            "Left Lung": masks['left_lung'],
            "Right Lung": masks['right_lung'],
            "Heart": masks['heart'],
            "Height": padding_info["height"],
            "Width": padding_info["width"]
        }
    

    def process(self, output_path: Path) -> None:
        """Process all images and save results to CSV."""
        processed_data = []
        
        for _, row in tqdm(
            self.annotations_df.iterrows(), 
            total=len(self.annotations_df),
            desc="Processing landmarks"
        ):
            print(self.padding_df[self.padding_df.filename == row["image_id"]])
            padding_info = self.padding_df[self.padding_df.filename == row["image_id"]].iloc[0]
            
            landmarks = np.array(eval(row["Landmarks"]))
            processed_landmarks = self._process_landmarks(
                landmarks,
                padding_info["height"],
                padding_info["width"],
                padding_info["pad_left"],
                padding_info["pad_top"]
            )
            
            masks = self._create_masks(processed_landmarks, (padding_info["height"], padding_info["width"]))
            
            processed_data.append(self._create_row(row, padding_info, masks))
        
        # Create and save the processed DataFrame
        pd.DataFrame(processed_data).to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


def main():
    # base_path = Path("../Annotations")
    
    # processor = LandmarkProcessor(
    #     annotations_path=base_path / "Preprocessed/VinDr-CXR.csv",
    #     padding_path=Path("path/to/paddings.csv")  # Update with actual path
    # )
    
    # processor.process(
    #     output_path=base_path / "OriginalResolution/VinDr-CXR.csv"
    # )

    base_path = Path(r"E:\Kai_2\DATA_Set\X-ray\CheXmask")
    
    processor = LandmarkProcessor(
        annotations_path=base_path / "Preprocessed/VinDr-CXR.csv",
        padding_path=Path(r"E:\Kai_2\DATA_Set\X-ray\VinDr-CXR\png_paddings.csv")  # Update with actual path
    )
    
    processor.process(
        output_path=base_path / "OriginalResolution/VinDr-CXRv2.csv"
    )


if __name__ == "__main__":
    main()