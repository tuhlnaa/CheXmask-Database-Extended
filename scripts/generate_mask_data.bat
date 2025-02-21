@echo off
setlocal

rem Configuration
set "CSV_PATH=.\data\processed_annotations_VinDr-CXR.csv"
set "OUTPUT_DIR=.\VinDr-CXR\mask"
set IMAGE_DIR=".\VinDr-CXR\train_png" ".\VinDr-CXR\test_png"

set SAVE_OPTIONS="left_lung" "right_lung" "heart" "combination"

rem Print header
echo === CheXmask Dataset Processing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the processing script
echo [EXECUTING] Starting CheXmask processing...
python utils\generate_mask_data.py ^
    --csv_path "%CSV_PATH%" ^
    --image_dir %IMAGE_DIR% ^
    --output_dir "%OUTPUT_DIR%" ^
    --save_options %SAVE_OPTIONS% ^
    --use_original
    
echo === Processing completed at %date% %time% ===

rem Display output directory location
echo Visualizations saved to: %OUTPUT_DIR%

pause
endlocal