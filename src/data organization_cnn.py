import os
import shutil
import random

# Set the random seed for reproducibility
random.seed(42)

# Source directories
source_dirs = {
    'original': 'dataset/first_print/First Print',
    'counterfeit': 'dataset/second_print/Second Print'
}

# Destination base directory
dest_base = 'dataset_cnn'
train_ratio = 0.8  # 80% for training

# Create destination directories
for label in source_dirs.keys():
    train_dir = os.path.join(dest_base, 'train', label)
    val_dir = os.path.join(dest_base, 'validation', label)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

# Split and copy images
for label, src in source_dirs.items():
    # Get list of image files from the source directory
    files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    random.shuffle(files)
    
    # Calculate split index
    train_count = int(len(files) * train_ratio)
    train_files = files[:train_count]
    val_files = files[train_count:]
    
    # Copy files to train folder
    for f in train_files:
        src_path = os.path.join(src, f)
        dst_path = os.path.join(dest_base, 'train', label, f)
        shutil.copy(src_path, dst_path)
    
    # Copy files to validation folder
    for f in val_files:
        src_path = os.path.join(src, f)
        dst_path = os.path.join(dest_base, 'validation', label, f)
        shutil.copy(src_path, dst_path)

print("Dataset organized successfully!")
