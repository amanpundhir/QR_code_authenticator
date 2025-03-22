from skimage.feature import local_binary_pattern, hog

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
        # Resize the image for consistency
        img_resized = cv2.resize(img, (128, 128))
        lbp_feat = extract_lbp_features(img_resized)
        hog_feat = extract_hog_features(img_resized)
        combined = np.hstack([lbp_feat, hog_feat])
        feature_list.append(combined)
    return np.array(feature_list)

# Extract features for both classes
features_first = extract_features(first_images)
features_second = extract_features(second_images)

# Create labels (0 for original, 1 for counterfeit)
labels_first = np.zeros(len(features_first))
labels_second = np.ones(len(features_second))

# Combine features and labels
X = np.vstack([features_first, features_second])
y = np.hstack([labels_first, labels_second])
print("Feature matrix shape:", X.shape)
