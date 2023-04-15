# split dataset into train:val:test = 75:5:20

import os
import random

train_path = "./train"
val_path = "./val"
test_path = "./test"

image_folder = "images"
text_folder = "mangatxt"

def split_val_test():
    train_image_files = [f for f in os.listdir(f'{train_path}/{image_folder}') if f.endswith('.jpg')]
    train_image_len = len(train_image_files)
    
    val_test_indices = random.sample(range(train_image_len), int(0.25 * train_image_len))
    val_len = int(5 / 20 * len(val_test_indices))
    val_indices, test_indices = val_test_indices[:val_len], val_test_indices[val_len:]
    
    # move images
    os.makedirs(f'{val_path}/{image_folder}', exist_ok=True)
    os.makedirs(f'{test_path}/{image_folder}', exist_ok=True)
    for val_idx in val_indices:
        img_name = train_image_files[val_idx]
        os.system(f'mv "{train_path}/{image_folder}/{img_name}" "{val_path}/{image_folder}/{img_name}"')
    
    for test_idx in test_indices:
        img_name = train_image_files[test_idx]
        os.system(f'mv "{train_path}/{image_folder}/{img_name}" "{test_path}/{image_folder}/{img_name}"')
    
    # move annotations
    os.makedirs(f'{val_path}/{text_folder}', exist_ok=True)
    os.makedirs(f'{test_path}/{text_folder}', exist_ok=True)
    for val_idx in val_indices:
        # MARS_001.jpg -> MARS_001.txt
        txt_name = train_image_files[val_idx][:-3] + "txt"
        os.system(f'mv "{train_path}/{text_folder}/{txt_name}" "{val_path}/{text_folder}/{txt_name}"')
    
    for test_idx in test_indices:
        txt_name = train_image_files[test_idx][:-3] + "txt"
        os.system(f'mv "{train_path}/{text_folder}/{txt_name}" "{test_path}/{text_folder}/{txt_name}"')
    
    
random.seed(42)
split_val_test()