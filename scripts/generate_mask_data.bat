@echo off
setlocal

rem Configuration
set "CSV_PATH=.\data\chexmask_annotations.csv"
set "OUTPUT_DIR=.\output"
set SAVE_OPTIONS="left_lung" "right_lung" "heart" "combination"

rem Print header
echo === CheXmask Dataset Processing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the processing script
echo [EXECUTING] Starting CheXmask processing...
python utils\generate_mask_data.py ^
    --csv_path "%CSV_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --save_options %SAVE_OPTIONS%

echo === Processing completed at %date% %time% ===

rem Display output directory location
echo Visualizations saved to: %OUTPUT_DIR%

pause
endlocal