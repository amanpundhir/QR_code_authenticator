import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Function to load grayscale images from a folder
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

# Set directories for original and counterfeit images
first_dir = 'dataset/first_print/First Print'
second_dir = 'dataset/second_print/Second Print'

# Load images
first_images, _ = load_images_from_folder(first_dir)
second_images, _ = load_images_from_folder(second_dir)
print("Original images loaded:", len(first_images))
print("Counterfeit images loaded:", len(second_images))

# Feature extraction functions
def extract_lbp_features(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def extract_features(images):
    feature_list = []
    for img in images:
        # Resize image for consistency
        img_resized = cv2.resize(img, (128, 128))
        lbp_feat = extract_lbp_features(img_resized)
        hog_feat = extract_hog_features(img_resized)
        combined = np.hstack([lbp_feat, hog_feat])
        feature_list.append(combined)
    return np.array(feature_list)

# Extract features and create labels (0 = original, 1 = counterfeit)
features_first = extract_features(first_images)
features_second = extract_features(second_images)
labels_first = np.zeros(len(features_first))
labels_second = np.ones(len(features_second))

# Combine data and labels
X = np.vstack([features_first, features_second])
y = np.hstack([labels_first, labels_second])
print("Feature matrix shape:", X.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest (Traditional CV) Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
