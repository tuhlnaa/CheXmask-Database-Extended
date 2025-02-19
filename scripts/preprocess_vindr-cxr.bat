@echo off
setlocal

rem Configuration
set "INPUT_DIR=.\data\train"
set "OUTPUT_DIR=.\data\train_png_1024"
set "TARGET_SIZE=1024"
set "NUM-PROCESSES=10"
set "LOG_LEVEL=INFO"

rem Print header
echo === DICOM X-Ray Image Processing Pipeline ===
echo Starting execution at %date% %time%

rem Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

rem Execute the preprocessing script
echo [EXECUTING] Starting DICOM processing...
python DataPreparation\vindr-cxr.py ^
    --input-dir "%INPUT_DIR%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --target-size %TARGET_SIZE% ^
    --num-processes %NUM-PROCESSES% ^
    --log-level %LOG_LEVEL%

set "OUTPUT_DIR=E:\Kai_2\DATA_Set\X-ray\VinDr-CXR\train_png"

python DataPreparation\vindr-cxr.py ^
    --input-dir "%INPUT_DIR%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --target-size %TARGET_SIZE% ^
    --num-processes %NUM-PROCESSES% ^
    --log-level %LOG_LEVEL% ^
    --preserve-resolution

echo === Processing completed at %date% %time% ===

rem Display metadata file location
echo Metadata saved to: %OUTPUT_DIR%\image_paddings.csv

pause
endlocal