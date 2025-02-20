import sys
import cv2
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils import decode_rle_to_mask


@dataclass
class LandmarkPoints:
    """Container for organ landmark points."""
    right_lung: np.ndarray
    left_lung: np.ndarray
    heart: np.ndarray


@dataclass
class OrganMasks:
    """Container for organ segmentation masks."""
    right_lung: np.ndarray
    left_lung: np.ndarray
    heart: np.ndarray


class CheXmaskProcessor:
    """Process and visualize CheXmask dataset annotations."""
    
    def __init__(self, csv_path: str, image_dir: Optional[str] = None):
        """
        Initialize the CheXmask processor.
        
        Args:
            csv_path: Path to the CheXmask CSV annotations file
            image_dir: Optional path to directory containing original images
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir) if image_dir else None


    def _parse_landmarks(self, landmark_str: str) -> np.ndarray:
        """Parse landmark string into numpy array."""
        try:
            landmarks = eval(landmark_str)
        except:
            # Handle different Python version formats
            landmarks = landmark_str.replace('[ ', '[').replace('\n ', ',') \
                                  .replace('  ', ',').replace(' ', ',')
            landmarks = eval(landmarks)

        return np.array(landmarks).reshape(-1, 2)


    def _split_landmarks(self, landmarks: np.ndarray) -> LandmarkPoints:
        """Split landmarks array into organ-specific points."""
        return LandmarkPoints(
            right_lung=landmarks[:44, :],
            left_lung=landmarks[44:94, :],
            heart=landmarks[94:, :]
        )


    def _load_original_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load the original image if available."""
        if not self.image_dir:
            return None
            
        image_path = self.image_dir / f"{image_id}.png"
        if not image_path.exists():
            print(f"Warning: Original image not found at {image_path}")
            return None
            
        return cv2.imread(str(image_path))


    def _create_colored_mask(self, masks: OrganMasks, dimensions: Tuple[int, int], 
                           original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Create a colored combination of all organ masks."""
        height, width = dimensions
        
        if original_image is not None:
            # Resize original image if dimensions don't match
            if original_image.shape[:2] != (height, width):
                original_image = cv2.resize(original_image, (width, height))
            base_img = original_image.copy()
        else:
            base_img = np.zeros([height, width, 3], dtype=np.uint8)
        
        # Create colored overlay
        overlay = np.zeros_like(base_img)
        overlay[:, :, 0] = masks.right_lung
        overlay[:, :, 1] = masks.left_lung
        overlay[:, :, 2] = masks.heart

        # Blend with original image if using it
        if original_image is not None:
            alpha = 0.5  # Adjust transparency
            mask = (overlay > 0).any(axis=2, keepdims=True)
            result = np.where(mask, cv2.addWeighted(base_img, 1-alpha, overlay, alpha, 0), base_img)
        else:
            result = overlay
            
        return result


    def _draw_landmarks(self, img: np.ndarray, landmarks: LandmarkPoints, 
                       colors: Optional[Tuple[Tuple[int, int, int], ...]] = None) -> np.ndarray:
        """Draw landmarks on the image."""
        if colors is None:
            colors = ((255, 0, 255),  # Right lung (magenta)
                     (255, 0, 255),  # Left lung (magenta)
                     (255, 255, 0))  # Heart (yellow)
        
        result = img.copy()
        for points, color in zip([landmarks.right_lung, landmarks.left_lung, landmarks.heart], colors):
            for x, y in points.astype(int):
                cv2.circle(result, (x, y), 10, color, -1, lineType=cv2.LINE_AA)
        return result


    def process_single_image(self, row_data: Dict, output_dir: str, 
                           save_options: List[str], use_original: bool) -> None:
        """Process a single image with all its visualizations."""
        # Parse landmarks
        landmarks = self._parse_landmarks(row_data["Landmarks"])
        current_landmarks = self._split_landmarks(landmarks)
        
        # Get image dimensions
        height, width = int(row_data["Height"]), int(row_data["Width"])

        # Get masks
        current_masks = OrganMasks(
            right_lung=decode_rle_to_mask(row_data["Right Lung"], height, width),
            left_lung=decode_rle_to_mask(row_data["Left Lung"], height, width),
            heart=decode_rle_to_mask(row_data["Heart"], height, width)
        )
        
        # Load original image if needed
        current_image = self._load_original_image(row_data['image_id']) if use_original else None

        # Create base image for landmarks
        if use_original and current_image is not None:
            base_img = current_image.copy()
        else:
            base_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Define visualization options
        visualization_map = {
            'left_lung': lambda: (current_masks.left_lung * 255).astype(np.uint8),
            'right_lung': lambda: (current_masks.right_lung * 255).astype(np.uint8),
            'heart': lambda: (current_masks.heart * 255).astype(np.uint8),
            'landmarks': lambda: self._draw_landmarks(base_img, current_landmarks),
            'combination': lambda: self._draw_landmarks(
                self._create_colored_mask(current_masks, (height, width), current_image),
                current_landmarks
            )
        }
        
        # Save visualizations
        output_path = Path(output_dir)
        for option in save_options:
            if option not in visualization_map:
                print(f"Warning: Unknown save option '{option}'")
                continue
                
            save_dir = output_path / option
            save_dir.mkdir(parents=True, exist_ok=True)
            
            visualization = visualization_map[option]()
            cv2.imwrite(str(save_dir / f"{row_data['image_id']}.png"), visualization)


def process_wrapper(args):
    """Wrapper function for parallel processing."""
    processor, row_data, output_dir, save_options, use_original = args
    try:
        processor.process_single_image(row_data, output_dir, save_options, use_original)
        return True
    except Exception as e:
        print(f"Error processing image {row_data['image_id']}: {str(e)}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process CheXmask dataset annotations.')
    parser.add_argument('--csv_path', type=str, required=True,
                      help='Path to the CheXmask CSV annotations file')
    parser.add_argument('--image_dir', type=str, help='Path to directory containing original images')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save output visualizations')
    parser.add_argument('--save_options', nargs='+', 
                      default=['left_lung', 'right_lung', 'heart', 'landmarks', 'combination'],
                      help='List of visualization options to save')
    parser.add_argument('--use_original', action='store_true',
                      help='Use original images as background when available')
    parser.add_argument('--num_workers', type=int, default=10,
                      help='Number of worker processes to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize processor
    processor = CheXmaskProcessor(args.csv_path, args.image_dir)
    
    # Prepare arguments for parallel processing
    process_args = [
        (processor, row, args.output_dir, args.save_options, args.use_original)
        for _, row in processor.df.iterrows()
    ]
    
    # Process images in parallel with progress bar
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(process_wrapper, process_args),
            total=len(process_args),
            desc="Processing images"
        ))
    
    # Report completion statistics
    successful = sum(results)
    total = len(results)
    print(f"\nProcessing complete: {successful}/{total} images processed successfully")

if __name__ == "__main__":
    main()