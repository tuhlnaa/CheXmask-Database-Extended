"""
Module for processing DICOM X-ray images and converting them to PNG format.
Includes functionality for padding images to square dimensions and saving metadata.
Enhanced with multiprocessing and tqdm progress bars.
"""
import cv2
import pydicom
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pydicom.pixel_data_handlers.util import apply_voi_lut
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dicom_processor")


@dataclass
class PaddingMetadata:
    """Store padding information for image transformations."""
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


class DicomProcessor:
    """Process DICOM images with padding and resizing capabilities."""
    
    def __init__(self, input_dir: Path, output_dir: Path, target_size: Optional[int] = 1024, 
                 preserve_resolution: bool = False):
        """
        Initialize the DICOM processor.
        
        Args:
            input_dir: Directory containing DICOM files
            output_dir: Directory for output PNG files
            target_size: Target size for the output square images (ignored if preserve_resolution is True)
            preserve_resolution: If True, maintain original resolution without padding or resizing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.preserve_resolution = preserve_resolution
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def read_xray(self, path: Path, voi_lut: bool = True, 
                  fix_monochrome: bool = True) -> Tuple[np.ndarray, Optional[float]]:
        """Read and process a DICOM X-ray image."""
        dicom = pydicom.dcmread(path)
        
        # Extract pixel spacing if available
        pixel_spacing = None
        if hasattr(dicom, 'ImagerPixelSpacing'):
            pixel_spacing = float(dicom.ImagerPixelSpacing[0])
        elif hasattr(dicom, 'PixelSpacing'):
            pixel_spacing = float(dicom.PixelSpacing[0])
        
        # Apply VOI LUT for human-friendly viewing if available
        data = apply_voi_lut(dicom.pixel_array, dicom) if voi_lut else dicom.pixel_array
            
        # Fix inverted X-ray if necessary
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
            
        # Normalize to 8-bit range
        data = self._normalize_to_uint8(data)
        return data, pixel_spacing


    @staticmethod
    def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
        """Normalize image data to uint8 format."""
        data_min = np.min(data)
        data_range = np.max(data) - data_min
        normalized = ((data - data_min) / data_range * 255).astype(np.uint8)
        return normalized


    def pad_to_square(self, img: np.ndarray) -> Tuple[np.ndarray, PaddingMetadata]:
        """Pad image to square dimensions with centered content."""
        height, width = img.shape[:2]
        longest_edge = max(height, width)
        
        pad_h = (longest_edge - height) // 2
        pad_w = (longest_edge - width) // 2
        
        result = np.zeros((longest_edge, longest_edge), dtype=img.dtype)
        result[pad_h:pad_h + height, pad_w:pad_w + width] = img
        
        padding = PaddingMetadata(
            pad_left=pad_w,
            pad_top=pad_h,
            pad_right=longest_edge - width - pad_w,
            pad_bottom=longest_edge - height - pad_h
        )
        
        return result, padding


    def find_dicom_files(self) -> List[Path]:
        """Find all DICOM files in the input directory."""
        return list(self.input_dir.rglob("*.dicom"))


    def process_single_file(self, file_path: Path) -> Dict:
        """Process a single DICOM file and return its metadata."""
        try:
            # Process image
            data, pixel_spacing = self.read_xray(file_path)
            height, width = data.shape[:2]
            
            if self.preserve_resolution:
                # Keep original image without padding or resizing
                output_img = data
                padding = None
            else:
                # Pad and resize as before
                output_img, padding = self.pad_to_square(data)
                output_img = cv2.resize(output_img, (self.target_size, self.target_size))
            
            # Save processed image
            relative_path = file_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path.with_suffix('.png')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output_img)
            
            # Calculate scaled pixel spacing only if resizing was done
            scaled_pixel_spacing = None
            if pixel_spacing is not None and not self.preserve_resolution:
                scale_factor = self.target_size / max(height, width)
                scaled_pixel_spacing = pixel_spacing / scale_factor
            
            # Return metadata
            metadata = {
                'filename': str(relative_path),
                'width': width,
                'height': height,
                'original_pixel_spacing_mm': pixel_spacing,
                'scaled_pixel_spacing_mm': scaled_pixel_spacing,
                'resolution_preserved': self.preserve_resolution
            }
            
            # Add padding information only if padding was performed
            if padding is not None:
                metadata.update(padding.__dict__)
                
            return metadata
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None


    def process_files(self, num_processes: int) -> pd.DataFrame:
        """
        Process all DICOM files in parallel and save as PNG images.
        
        Args:
            num_processes: Number of processes to use for parallel processing
            
        Returns:
            DataFrame containing metadata for all processed images
        """
        dicom_files = self.find_dicom_files()
        total_files = len(dicom_files)
        
        logger.info(f"Found {total_files} DICOM files to process")
        logger.info(f"Using {num_processes} processes for parallel processing")
        
        metadata = []
        # Use ProcessPoolExecutor for parallel processing with tqdm for progress tracking
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Create a partial function with the self argument bound
            process_func = partial(self.process_single_file)
            
            # Process files in parallel with progress bar
            results = list(tqdm(
                executor.map(process_func, dicom_files),
                total=total_files,
                desc="Processing DICOM files",
                unit="files"
            ))
            
            # Filter out None results (failed processing) and collect metadata
            metadata = [result for result in results if result is not None]
        
        return pd.DataFrame(metadata)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process DICOM X-ray images to PNG format with parallel processing."
    )
    parser.add_argument("--input-dir", type=Path, required=True, 
                       help="Directory containing DICOM files")
    parser.add_argument("--output-dir", type=Path, required=True, 
                       help="Directory for output PNG files")
    parser.add_argument("--target-size", type=int,
                       help="Target size for the output square images (required unless --preserve-resolution is used)")
    parser.add_argument("--preserve-resolution", action="store_true",
                       help="Preserve original resolution without padding or resizing")
    parser.add_argument("--num-processes", type=int, 
                       default=max(1, mp.cpu_count() - 1),
                       help="Number of processes to use for parallel processing (default: number of CPU cores - 1)")
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Set the logging level (default: INFO)")
    args = parser.parse_args()

    # Validate arguments
    if not args.preserve_resolution and args.target_size is None:
        parser.error("--target-size is required when --preserve-resolution is not used")

    return args


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set log level
    logger.setLevel(args.log_level)
    logger.info(f"Processing DICOM files from {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    if not args.preserve_resolution:
        logger.info(f"Target size: {args.target_size}x{args.target_size}")
    logger.info(f"Preserve resolution: {args.preserve_resolution}")
    
    processor = DicomProcessor(
        args.input_dir, 
        args.output_dir, 
        args.target_size if not args.preserve_resolution else None,
        args.preserve_resolution
    )
    metadata_df = processor.process_files(args.num_processes)
    
    # Save metadata
    metadata_path = args.output_dir.parents[0] / (args.output_dir.name + '_paddings.csv')
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Successfully processed {len(metadata_df)} files")
    logger.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()