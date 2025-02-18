import sys
import cv2
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from DataPostprocessing.utils import get_mask_from_RLE

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
    

    def load_example(self, index: int = 0) -> None:
        """
        Load a specific example from the dataset.
        
        Args:
            index: Index of the example to load
        """
        self.current_example = self.df.iloc[index]
        
        # Parse landmarks
        landmarks = self._parse_landmarks(self.current_example["Landmarks"])
        self.current_landmarks = self._split_landmarks(landmarks)
        
        # Get image dimensions
        height, width = self.current_example["Height"], self.current_example["Width"]
        
        # Get masks
        self.current_masks = OrganMasks(
            right_lung=get_mask_from_RLE(self.current_example["Right Lung"], height, width),
            left_lung=get_mask_from_RLE(self.current_example["Left Lung"], height, width),
            heart=get_mask_from_RLE(self.current_example["Heart"], height, width)
        )
    

    def _create_colored_mask(self) -> np.ndarray:
        """Create a colored combination of all organ masks."""
        height, width = self.current_example["Height"], self.current_example["Width"]
        mask = np.zeros([height, width, 3], dtype=np.uint8)
        mask[:, :, 0] = self.current_masks.right_lung
        mask[:, :, 1] = self.current_masks.left_lung
        mask[:, :, 2] = self.current_masks.heart
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
    

    def save_visualizations(self, output_dir: str = "output", save_options: List[str] = None) -> None:
        """
        Save visualizations based on specified options.
        
        Args:
            output_dir: Directory to save the output images
            save_options: List of options to save. Valid options are:
                         ['left_lung', 'right_lung', 'heart', 'landmarks', 'combination']
        """
        if self.current_example is None:
            raise ValueError("No example loaded. Call load_example() first.")
            
        if save_options is None:
            save_options = ['combination']
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        height, width = self.current_example["Height"], self.current_example["Width"]
        
        for option in save_options:
            if option == 'left_lung':
                print(self.current_masks.left_lung.shape)
                cv2.imwrite(str(output_path / 'left_lung_mask.png'),
                          self.current_masks.left_lung)
                
            elif option == 'right_lung':
                cv2.imwrite(str(output_path / 'right_lung_mask.png'),
                          self.current_masks.right_lung)
                
            elif option == 'heart':
                cv2.imwrite(str(output_path / 'heart_mask.png'),
                          self.current_masks.heart)
                
            elif option == 'landmarks':
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                result = self._draw_landmarks(blank, self.current_landmarks)
                cv2.imwrite(str(output_path / 'landmarks.png'), result)
                
            elif option == 'combination':
                colored_mask = self._create_colored_mask()
                result = self._draw_landmarks(colored_mask, self.current_landmarks)
                cv2.imwrite(str(output_path / 'combination.png'), result)
                
            else:
                print(f"Warning: Unknown save option '{option}'")


def main():
    """Example usage of the CheXmask processor."""
    #processor = CheXmaskProcessor("Annotations/OriginalResolution/VinDr-CXR.csv.gz")
    processor = CheXmaskProcessor(r"E:\Kai_2\DATA_Set\X-ray\CheXmask\Annotations\OriginalResolution\VinDr-CXR.csv.gz")
    # Load the first example
    processor.load_example(0)
    
    # Save all visualization options
    processor.save_visualizations(
        save_options=['left_lung', 'right_lung', 'heart', 'landmarks', 'combination']
    )


if __name__ == "__main__":
    main()