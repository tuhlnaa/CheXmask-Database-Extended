import pandas as pd

def merge_csv_files(file1_path, file2_path, output_path):
    """
    Merge two CSV files and export the combined data to a new CSV file.
    
    Parameters:
    file1_path (str): Path to the first CSV file
    file2_path (str): Path to the second CSV file
    output_path (str): Path where the merged CSV will be saved
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Merge the dataframes and drop duplicates if any
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Drop duplicates based on first column
        merged_df = merged_df.drop_duplicates(subset=[merged_df.columns[0]], keep='first')
        
        # Sort by first column
        merged_df = merged_df.sort_values(merged_df.columns[0])
        
        # Export to CSV
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully merged files. Output saved to: {output_path}")
        print(f"Total rows in merged file: {len(merged_df)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    file1_path = "paddings1.csv"
    file2_path = "paddings2.csv"
    output_path = "merged_paddings.csv"

    file1_path = r"E:\Kai_2\DATA_Set\X-ray\VinDr-CXR\test_png_paddings.csv"
    file2_path = r"E:\Kai_2\DATA_Set\X-ray\VinDr-CXR\train_png_paddings.csv"
    output_path = r"E:\Kai_2\DATA_Set\X-ray\VinDr-CXR\png_paddings.csv"

    merge_csv_files(file1_path, file2_path, output_path)