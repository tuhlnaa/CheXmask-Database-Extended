"""
Utilities for processing and visualizing CheXmask dataset annotations.

The CheXmask dataset contains chest X-ray annotations with the following data:
- Organ segmentation masks (lungs and heart) in RLE format
- Landmark points for organ contours
- Image metadata (dimensions, quality metrics)
"""

from typing import Tuple, List, Dict, Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from dataclasses import dataclass

@dataclass
class OrganLandmarks:
    """Container for organ-specific landmarks."""
    right_lung: np.ndarray  # Shape: (44, 2)
    left_lung: np.ndarray   # Shape: (50, 2)
    heart: np.ndarray       # Shape: (N, 2)

class ChexMaskProcessor:
    """Processor for CheXmask dataset annotations."""
    
    def __init__(self, annotations_path: Union[str, Path]):
        """
        Initialize the processor with path to annotations.
        
        Args:
            annotations_path: Path to the .csv.gz annotations file
        """
        self.df = pd.read_csv(annotations_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        
    def parse_landmarks(self, landmarks_str: str) -> np.ndarray:
        """
        Parse landmarks string into numpy array.
        
        Args:
            landmarks_str: String containing landmark coordinates
            
        Returns:
            Array of shape (N, 2) containing landmark coordinates
        """
        try:
            landmarks = eval(landmarks_str)
        except:
            # Handle alternative format
            landmarks = (landmarks_str.replace('[ ', '[')
                                   .replace('\n ', ',')
                                   .replace('  ', ',')
                                   .replace(' ', ','))
            landmarks = eval(landmarks)
        
        return np.array(landmarks).reshape(-1, 2)
    
    def get_sample_landmarks(self, idx: int = 0) -> OrganLandmarks:
        """
        Get landmarks for a specific sample.
        
        Args:
            idx: Index of the sample in the dataset
            
        Returns:
            OrganLandmarks object containing separated landmarks for each organ
        """
        example = self.df.iloc[idx]
        landmarks = self.parse_landmarks(example["Landmarks"])
        
        return OrganLandmarks(
            right_lung=landmarks[:44],
            left_lung=landmarks[44:94],
            heart=landmarks[94:]
        )
    
    def get_sample_dimensions(self, idx: int = 0) -> Tuple[int, int]:
        """Get height and width of a specific sample."""
        example = self.df.iloc[idx]
        return int(example["Height"]), int(example["Width"])

class ChexMaskVisualizer:
    """Visualization utilities for CheXmask data."""
    
    @staticmethod
    def plot_landmarks(landmarks: OrganLandmarks, height: int, width: int) -> None:
        """Plot organ landmarks with different colors."""
        plt.figure(figsize=(10, 10))
        
        plt.scatter(landmarks.right_lung[:, 0], landmarks.right_lung[:, 1], c="r", label="Right Lung")
        plt.scatter(landmarks.left_lung[:, 0], landmarks.left_lung[:, 1], c="b", label="Left Lung")
        plt.scatter(landmarks.heart[:, 0], landmarks.heart[:, 1], c="g", label="Heart")
        
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Invert y-axis for image coordinates
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()
    
    @staticmethod
    def create_organ_overlay(
        image: np.ndarray,
        landmarks: OrganLandmarks,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create visualization overlay of organs on the input image.
        
        Args:
            image: Input grayscale image
            landmarks: OrganLandmarks object containing organ landmarks
            original_shape: Tuple of (height, width)
            
        Returns:
            RGB image with organ overlays
        """
        h, w = original_shape
        dense_mask = ChexMaskVisualizer._get_dense_mask(landmarks, h, w)
        
        # Create RGB overlay
        overlay = np.zeros([h, w, 3])
        overlay[:,:,0] = (image + 
                         0.3 * (dense_mask == 1).astype(float) - 
                         0.1 * (dense_mask == 2).astype(float))
        overlay[:,:,1] = (image + 
                         0.3 * (dense_mask == 2).astype(float) - 
                         0.1 * (dense_mask == 1).astype(float))
        overlay[:,:,2] = (image - 
                         0.1 * (dense_mask == 1).astype(float) - 
                         0.2 * (dense_mask == 2).astype(float))
        
        overlay = np.clip(overlay, 0, 1)
        
        # Add landmark points
        for organ, color in [
            (landmarks.right_lung, (1, 0, 1)),
            (landmarks.left_lung, (1, 0, 1)),
            (landmarks.heart, (1, 1, 0))
        ]:
            for point in organ:
                cv2.circle(
                    overlay,
                    (int(point[0]), int(point[1])),
                    5,
                    color,
                    -1
                )
        
        return overlay
    
    @staticmethod
    def _get_dense_mask(
        landmarks: OrganLandmarks,
        height: int,
        width: int
    ) -> np.ndarray:
        """Create dense mask from landmarks."""
        mask = np.zeros([height, width], dtype='uint8')
        
        # Reshape landmarks for cv2.drawContours
        organs = [
            (landmarks.right_lung, 1),
            (landmarks.left_lung, 1),
            (landmarks.heart, 2)
        ]
        
        for points, value in organs:
            contour = points.reshape(-1, 1, 2).astype('int')
            cv2.drawContours(mask, [contour], -1, value, -1)
        
        return mask


def load_example_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load example image and return with its shape."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    return img, img.shape


# Usage example:
if __name__ == "__main__":
    # Initialize processor
    processor = ChexMaskProcessor("Annotations/OriginalResolution/VinDr-CXR.csv.gz")
    
    # Get sample data
    landmarks = processor.get_sample_landmarks(0)
    height, width = processor.get_sample_dimensions(0)
    
    # Visualize landmarks
    visualizer = ChexMaskVisualizer()
    visualizer.plot_landmarks(landmarks, height, width)
    
    # Load and process example image
    img, shape = load_example_image("Example/utils_example1.jpg")
    overlay = visualizer.create_organ_overlay(img, landmarks, shape)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title("Organ Segmentation Overlay")
    plt.show()