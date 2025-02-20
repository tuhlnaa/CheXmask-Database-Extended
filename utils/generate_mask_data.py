import sys
import cv2
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
        self.current_example = None
        self.current_landmarks = None
        self.current_masks = None
        self.current_image = None


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


    def _load_original_image(self) -> Optional[np.ndarray]:
        """Load the original image if available."""
        if not self.image_dir:
            return None
            
        image_path = self.image_dir / f"{self.current_example['image_id']}.png"
        if not image_path.exists():
            print(f"Warning: Original image not found at {image_path}")
            return None
            
        return cv2.imread(str(image_path))


    def load_example(self, index: int = 0) -> None:
        """Load a specific example from the dataset."""
        self.current_example = self.df.iloc[index]
        
        # Parse landmarks
        landmarks = self._parse_landmarks(self.current_example["Landmarks"])
        self.current_landmarks = self._split_landmarks(landmarks)
        
        # Get image dimensions
        height, width = int(self.current_example["Height"]), int(self.current_example["Width"])

        # Get masks
        self.current_masks = OrganMasks(
            right_lung=decode_rle_to_mask(self.current_example["Right Lung"], height, width),
            left_lung=decode_rle_to_mask(self.current_example["Left Lung"], height, width),
            heart=decode_rle_to_mask(self.current_example["Heart"], height, width)
        )
        
        # Load original image if available
        self.current_image = self._load_original_image()


    def _create_colored_mask(self, use_original: bool = False) -> np.ndarray:
        """
        Create a colored combination of all organ masks.
        
        Args:
            use_original: If True and original image is available, use it as background
        """
        height, width = int(self.current_example["Height"]), int(self.current_example["Width"])
        
        if use_original and self.current_image is not None:
            # Resize original image if dimensions don't match
            if self.current_image.shape[:2] != (height, width):
                self.current_image = cv2.resize(self.current_image, (width, height))
            base_img = self.current_image.copy()
        else:
            base_img = np.zeros([height, width, 3], dtype=np.uint8)
        
        # Create colored overlay
        overlay = np.zeros_like(base_img)
        overlay[:, :, 0] = self.current_masks.right_lung  # Pixel Value=255
        overlay[:, :, 1] = self.current_masks.left_lung
        overlay[:, :, 2] = self.current_masks.heart

        # Blend with original image if using it
        if use_original and self.current_image is not None:
            alpha = 0.5  # Adjust transparency
            mask = (overlay > 0).any(axis=2, keepdims=True)
            result = np.where(mask, cv2.addWeighted(base_img, 1-alpha, overlay, alpha, 0), base_img)
        else:
            result = overlay
            
        return result


    def _draw_landmarks(
            self, 
            img: np.ndarray, 
            landmarks: LandmarkPoints, 
            colors: Optional[Tuple[Tuple[int, int, int], ...]] = None
        ) -> np.ndarray:
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


    def save_visualizations(
            self, 
            output_dir: str = "output", 
            save_options: List[str] = None,
            use_original: bool = False
        ) -> None:
        """
        Save visualizations based on specified options.
        
        Args:
            output_dir: Directory to save the output images
            save_options: List of options to save. Valid options are:
                        ['left_lung', 'right_lung', 'heart', 'landmarks', 'combination']
            use_original: If True and original images are available, use them as background
        """
        if self.current_example is None:
            raise ValueError("No example loaded. Call load_example() first.")
            
        save_options = save_options or ['combination']
        output_path = Path(output_dir)
        
        # Create base image for landmarks
        if use_original and self.current_image is not None:
            base_img = self.current_image.copy()
        else:
            base_img = np.zeros((
                int(self.current_example["Height"]), 
                int(self.current_example["Width"]), 
                3
            ), dtype=np.uint8)
        
        # Define mapping of options to their corresponding mask/visualization functions
        visualization_map = {
            'left_lung': lambda: (self.current_masks.left_lung * 255).astype(np.uint8),
            'right_lung': lambda: (self.current_masks.right_lung * 255).astype(np.uint8),
            'heart': lambda: (self.current_masks.heart * 255).astype(np.uint8),
            'landmarks': lambda: self._draw_landmarks(base_img, self.current_landmarks),
            'combination': lambda: self._draw_landmarks(
                self._create_colored_mask(use_original),
                self.current_landmarks
            )
        }
        
        for option in save_options:
            if option not in visualization_map:
                print(f"Warning: Unknown save option '{option}'")
                continue
                
            # Create subdirectory for each option
            save_dir = output_path / option
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save the visualization
            visualization = visualization_map[option]()
            cv2.imwrite(str(save_dir / f"{self.current_example['image_id']}.png"), visualization)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process CheXmask dataset annotations.')
    parser.add_argument('--csv_path', type=str, help='Path to the CheXmask CSV annotations file')
    parser.add_argument('--image_dir', type=str, help='Path to directory containing original images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output visualizations')
    parser.add_argument('--save_options', nargs='+', 
                      default=['left_lung', 'right_lung', 'heart', 'landmarks', 'combination'],
                      help='List of visualization options to save')
    parser.add_argument('--use_original', action='store_true',
                      help='Use original images as background when available')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize processor
    processor = CheXmaskProcessor(args.csv_path, args.image_dir)

    # Determine number of images to process
    total_images = len(processor.df)

    # Process images
    for idx in range(total_images):
        print(f"Processing image {idx + 1}/{total_images}")
        processor.load_example(idx)
        processor.save_visualizations(
            output_dir=args.output_dir,
            save_options=args.save_options,
            use_original=args.use_original
        )

if __name__ == "__main__":
    main()