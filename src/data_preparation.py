import os
import zipfile

# Define the zip file paths (update these paths to match your local files)
first_zip = 'First Print-20250322T093756Z-001.zip'
second_zip = 'Second Print-20250322T093757Z-001.zip'

# Define extraction directories
first_dir = 'dataset/first_print'
second_dir = 'dataset/second_print'

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

extract_zip(first_zip, first_dir)
extract_zip(second_zip, second_dir)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

# Load images from both classes
first_images, first_filenames = load_images_from_folder(first_dir)
second_images, second_filenames = load_images_from_folder(second_dir)

# Print basic statistics
print("Number of first prints (original):", len(first_images))
print("Number of second prints (counterfeit):", len(second_images))
print("Dataset distribution:", Counter(["original"] * len(first_images) + ["counterfeit"] * len(second_images)))

# Function to display a few sample images
def show_sample_images(images, title, num_samples=4):
    plt.figure(figsize=(10, 3))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"{title} {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_images(first_images, "Original")
show_sample_images(second_images, "Counterfeit")
