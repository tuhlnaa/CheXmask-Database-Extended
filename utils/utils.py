"""Utility functions for mask and landmark processing in medical image analysis.

This module provides functions for converting between RLE (Run Length Encoding)
and binary masks, as well as creating dense masks from landmarks.
"""

import cv2
import numpy as np
from typing import List, Tuple, Union


def encode_mask_to_rle(mask: np.ndarray) -> str:
    """Convert a binary mask to RLE (Run Length Encoding) format.

    Args:
        mask: Binary mask array with values 0 or 255

    Returns:
        str: Space-separated string of RLE encoded values
    """
    # Normalize mask to binary values (0 or 1)
    binary_mask = (mask / 255).astype(np.int32)
    pixels = binary_mask.flatten()
    
    # Add sentinel values for consistent run calculation
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Calculate runs by finding where values change
    run_starts = np.where(pixels[1:] != pixels[:-1])[0] + 1
    run_lengths = run_starts[1::2] - run_starts[::2]
    
    # Combine starts and lengths
    runs = np.stack([run_starts[::2], run_lengths], axis=1).flatten()
    return ' '.join(map(str, runs))


def decode_rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """Convert RLE (Run Length Encoding) string to binary mask.

    Args:
        rle: Space-separated string of RLE encoded values
        height: Height of the output mask
        width: Width of the output mask

    Returns:
        np.ndarray: Binary mask of shape (height, width) with values 0 or 255
    """
    runs = np.array([int(x) for x in rle.split()])
    
    # Separate starts and lengths
    starts = runs[::2] - 1  # Convert to 0-based indexing
    lengths = runs[1::2]
    
    # Create flattened mask
    mask = np.zeros(height * width, dtype=np.uint8)
    
    # Fill runs with 255
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 255
    
    return mask.reshape((height, width))


def create_dense_mask_from_landmarks(landmarks: np.ndarray, image_size: Union[int, Tuple[int, int]] = 1024) -> np.ndarray:
    """Create a binary mask from landmark points by filling their contour.

    Args:
        landmarks: Array of shape (N, 2) containing landmark coordinates
        image_size: Either a single integer for square images or (height, width) tuple

    Returns:
        np.ndarray: Binary mask with filled contour of landmarks
    """
    # Handle different image_size formats
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size
        
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Reshape landmarks for cv2.drawContours
    contour = landmarks.reshape(-1, 1, 2).astype(np.int32)
    
    # Fill contour with 255
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    return mask
