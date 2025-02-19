import pandas as pd 
import numpy as np 

from utils.utils import create_dense_mask_from_landmarks, encode_mask_to_rle 


path = "../Annotations/Preprocessed/VinDr-CXR.csv"
path2 = ... # path to paddings csv

df = pd.read_csv(path)
pads = pd.read_csv(path2)

pads.filename = pads.filename.str.replace(".dicom", "")
pads.filename = pads.filename.str.split("/").str[-1]

new_df = pd.DataFrame(columns=df.columns)

for index, row in df.iterrows():
    print(index)
    image_id = row["image_id"]
    
    pad_row = pads[pads.filename == image_id].iloc[0]
    
    height, width = pad_row["height"], pad_row["width"]
    
    landmarks = np.array(eval(row["Landmarks"])).reshape(-1, 2) / 1024
    max_shape = max(height, width)
    landmarks = landmarks * max_shape
    
    pad_left = pad_row["pad_left"]
    pad_top = pad_row["pad_top"]
    
    landmarks[:, 0] = landmarks[:, 0] - pad_left
    landmarks[:, 1] = landmarks[:, 1] - pad_top
    landmarks = np.round(landmarks).astype(int)
    
    RL = landmarks[:44]
    LL = landmarks[44:94]
    H = landmarks[94:]
    
    RL_ = create_dense_mask_from_landmarks(RL, (height, width))
    LL_ = create_dense_mask_from_landmarks(LL, (height, width))
    H_ = create_dense_mask_from_landmarks(H, (height, width))

    RL_RLE = encode_mask_to_rle(RL_)
    LL_RLE = encode_mask_to_rle(LL_)
    H_RLE = encode_mask_to_rle(H_)
    
    # columns = image_id Dice RCA (Mean)	Dice RCA (Max)	Landmarks	Left Lung	Right Lung	Heart	Height	Width
        
    new_row = {
        "image_id": row["image_id"],
        "Dice RCA (Mean)": row["Dice RCA (Mean)"],
        "Dice RCA (Max)": row["Dice RCA (Max)"],
        "Landmarks": landmarks,
        "Left Lung": LL_RLE,
        "Right Lung": RL_RLE,
        "Heart": H_RLE,
        "Height": height,
        "Width": width
    }
    
    new_df = new_df.append(new_row, ignore_index=True)

new_df.to_csv("../Annotations/OriginalResolution/VinDr-CXR.csv", index=False)