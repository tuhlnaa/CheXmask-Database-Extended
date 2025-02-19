import sys
import cv2
import numpy as np
import pandas as pd
import argparse

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
    
    def __init__(self, csv_path: str):
        """
        Initialize the CheXmask processor.
        
        Args:
            csv_path: Path to the CheXmask CSV annotations file
        """
        self.df = pd.read_csv(csv_path)
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
        """
        Split landmarks array into organ-specific points.
        
        Args:
            landmarks: Array of all landmark points
            
        Returns:
            LandmarkPoints object containing separated organ landmarks
        """
        return LandmarkPoints(
            right_lung=landmarks[:44, :],
            left_lung=landmarks[44:94, :],
            heart=landmarks[94:, :]
        )


    def load_example(self, index: int = 0, image_path: Optional[str] = None) -> None:
        """
        Load a specific example from the dataset.
        
        Args:
            index: Index of the example to load
            image_path: Optional path to the original image
        """
        self.current_example = self.df.iloc[index]
        
        # Parse landmarks
        landmarks = self._parse_landmarks(self.current_example["Landmarks"])
        self.current_landmarks = self._split_landmarks(landmarks)
        
        # Get image dimensions
        height, width = self.current_example["Height"], self.current_example["Width"]
        
        # Get masks (Convert RLE (Run Length Encoding) string to binary mask)
        self.current_masks = OrganMasks(
            right_lung=decode_rle_to_mask(self.current_example["Right Lung"], height, width),
            left_lung=decode_rle_to_mask(self.current_example["Left Lung"], height, width),
            heart=decode_rle_to_mask(self.current_example["Heart"], height, width)
        )

        # Load original image if provided
        self.current_image = None
        if image_path:
            self.current_image = cv2.imread(image_path)
            if self.current_image is not None:
                # Resize image to match mask dimensions if necessary
                if self.current_image.shape[:2] != (height, width):
                    self.current_image = cv2.resize(self.current_image, (width, height))


    def _create_colored_mask(self, use_original_image: bool = False) -> np.ndarray:
        """
        Create a colored combination of all organ masks.
        
        Args:
            use_original_image: Whether to use the original image as background
        """
        height, width = self.current_example["Height"], self.current_example["Width"]
        
        if use_original_image and self.current_image is not None:
            # Create a copy of the original image
            mask = self.current_image.copy()
            # Apply semi-transparent masks
            alpha = 0.5
            mask_overlay = np.zeros_like(mask)
            mask_overlay[:, :, 0] = self.current_masks.right_lung * 255
            mask_overlay[:, :, 1] = self.current_masks.left_lung * 255
            mask_overlay[:, :, 2] = self.current_masks.heart * 255
            mask = cv2.addWeighted(mask, 1 - alpha, mask_overlay, alpha, 0)
        else:
            # Create mask with black background
            mask = np.zeros([height, width, 3], dtype=np.uint8)
            mask[:, :, 0] = self.current_masks.right_lung * 255
            mask[:, :, 1] = self.current_masks.left_lung * 255
            mask[:, :, 2] = self.current_masks.heart * 255
        
        return mask


    def _draw_landmarks(self, img: np.ndarray, landmarks: LandmarkPoints,
                       colors: Optional[Tuple[Tuple[int, int, int], ...]] = None) -> np.ndarray:
        """
        Draw landmarks on the image.
        
        Args:
            img: Input image
            landmarks: LandmarkPoints object containing organ landmarks
            colors: Optional tuple of RGB colors for each organ's landmarks
            
        Returns:
            Image with drawn landmarks
        """
        if colors is None:
            colors = ((255, 0, 255),  # Right lung (magenta)
                     (255, 0, 255),  # Left lung (magenta)
                     (255, 255, 0))  # Heart (yellow)
        
        result = img.copy()
        for points, color in zip([landmarks.right_lung, landmarks.left_lung, landmarks.heart], colors):
            for x, y in points.astype(int):
                cv2.circle(result, (x, y), 10, color, -1, lineType=cv2.LINE_AA)
        return result


    def save_visualizations(self, output_dir: str = "output", save_options: List[str] = None, use_original_image: bool = False) -> None:
        """
        Save visualizations based on specified options.
        
        Args:
            output_dir: Directory to save the output images
            save_options: List of options to save. Valid options are:
                        ['left_lung', 'right_lung', 'heart', 'landmarks', 'combination']
            use_original_image: Whether to use the original image as background for
                              'landmarks' and 'combination' visualizations
        """
        if self.current_example is None:
            raise ValueError("No example loaded. Call load_example() first.")
            
        save_options = save_options or ['combination']
        output_path = Path(output_dir)
        
        background = self.current_image if use_original_image and self.current_image is not None else \
                    np.zeros((self.current_example["Height"], self.current_example["Width"], 3), dtype=np.uint8)
        
        # Define mapping of options to their corresponding mask/visualization functions
        visualization_map = {
            'left_lung': lambda: self.current_masks.left_lung * 255,
            'right_lung': lambda: self.current_masks.right_lung * 255,
            'heart': lambda: self.current_masks.heart * 255,
            'landmarks': lambda: self._draw_landmarks(background, self.current_landmarks),
            'combination': lambda: self._draw_landmarks(
                self._create_colored_mask(use_original_image),
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
    parser.add_argument('--csv_path', type=str, required=True,
                      help='Path to the CheXmask CSV annotations file')
    parser.add_argument('--image_dir', type=str,
                      help='Path to directory containing original images')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save output visualizations')
    parser.add_argument('--save_options', nargs='+', 
                      default=['left_lung', 'right_lung', 'heart', 'landmarks', 'combination'],
                      help='List of visualization options to save')
    parser.add_argument('--use_original_image', action='store_true',
                      help='Use original images as background for landmarks and combination visualizations')
    return parser.parse_args()


def main():
    """Main function to process CheXmask dataset."""
    args = parse_args()
    
    # Initialize processor
    processor = CheXmaskProcessor(args.csv_path)

    # Determine number of images to process
    total_images = len(processor.df)

    # Process images
    for idx in range(total_images):
        print(f"Processing image {idx + 1}/{total_images}")
        
        # Construct image path if image directory is provided
        image_path = None
        if args.image_dir:
            image_id = processor.df.iloc[idx]['image_id']
            image_path = str(Path(args.image_dir) / f"{image_id}.png")
        
        processor.load_example(idx, image_path)
        processor.save_visualizations(
            output_dir=args.output_dir,
            save_options=args.save_options,
            use_original_image=args.use_original_image
        )

if __name__ == "__main__":
    main()