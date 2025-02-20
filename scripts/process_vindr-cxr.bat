@echo off
setlocal

rem Configuration
set "ANNOTATIONS_PATH=.\data\VinDr-CXR.csv"
set "PADDING_PATH=.\data\png_paddings_1024.csv"
set "OUTPUT_PATH=.\output\processed_annotations_VinDr-CXR.csv"
set BATCH_SIZE=100
set MAX_WORKERS=16

rem Print header
echo === Chest X-ray Landmark Processing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the processing script
echo [EXECUTING] Starting landmark processing...
python DataPostprocessing\ReversePreprocessing\back_vindr-cxr.py ^
    --annotations-path "%ANNOTATIONS_PATH%" ^
    --padding-path "%PADDING_PATH%" ^
    --output-path "%OUTPUT_PATH%" ^
    --batch-size %BATCH_SIZE% ^
    --max-workers %MAX_WORKERS%

echo === Processing completed at %date% %time% ===

rem Display output file location
echo Processed data saved to: %OUTPUT_PATH%

pause
endlocal