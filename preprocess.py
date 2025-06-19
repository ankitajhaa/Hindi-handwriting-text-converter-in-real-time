import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# Path to the train directory
train_dir = 'dataset/train/'


# Image dimensions
img_size = 32


# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))  # Resize to expected input size
    img = img / 255.0  # Normalize to [0, 1]
    return img


# Load images and labels from the train directory
X = []
y = []

folder_to_char = {
    'character_1_ka': 'क',
    'character_2_kha': 'ख',
    'character_3_ga': 'ग',
    'character_4_gha': 'घ',
    'character_5_kna': 'ङ',
    'character_6_cha': 'च',
    'character_7_chha': 'छ',
    'character_8_ja': 'ज',
    'character_9_jha': 'झ',
    'character_10_yna': 'ञ',
    'character_11_ta': 'ट',
    'character_12_thha': 'ठ',
    'character_13_da': 'ड',
    'character_14_dha': 'ढ',
    'character_15_na': 'ण',
    'character_16_t': 'त',
    'character_17_tha': 'थ',
    'character_18_d': 'द',
    'character_19_dha': 'ध',
    'character_20_n': 'न',
    'character_21_pa': 'प',
    'character_22_fa': 'फ',
    'character_23_ba': 'ब',
    'character_24_bha': 'भ',
    'character_25_ma': 'म',
    'character_26_ya': 'य',
    'character_27_ra': 'र',
    'character_28_la': 'ल',
    'character_29_va': 'व',
    'character_30_sha': 'श',
    'character_31_shha': 'ष',
    'character_32_sa': 'स',
    'character_33_ha': 'ह',
    'character_34_ksha': 'क्ष',
    'character_35_tra': 'त्र',
    'character_36_gya': 'ज्ञ',
}


# Dynamically generate the list of characters (labels) from the folder names
class_names = os.listdir(train_dir)
class_names.sort()  # To ensure characters are sorted
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(train_dir, class_name)
   
    # Ensure it's a directory
    if os.path.isdir(class_folder):
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
           
            # Preprocess the image and append it to the data list
            img = preprocess_image(img_path)
            X.append(img)
            y.append(label)  # Assign label based on the folder name


# Convert to numpy arrays
X = np.array(X)
X = X.reshape(-1, img_size, img_size, 1)  # Reshape to include the channel dimension
y = np.array(y)


# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Save the preprocessed data for later use
np.save('data\X_train_devanagari.npy', X_train)
np.save('data\y_train_devanagari.npy', y_train)
np.save('data\X_test_devanagari.npy', X_test)
np.save('data\y_test_devanagari.npy', y_test)


print("Preprocessing completed!")