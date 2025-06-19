import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import pickle

# Ensure UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
MODEL_PATH = 'models/devanagari_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure the model is trained and saved correctly.")

model = tf.keras.models.load_model(MODEL_PATH)

# Load label mapping from pickle
LABEL_MAP_PATH = 'label_map.pkl'
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Label map file '{LABEL_MAP_PATH}' not found. Please ensure it was saved during training.")

with open(LABEL_MAP_PATH, 'rb') as f:
    folder_to_char = pickle.load(f)

# Ensure label list is properly ordered
characters = [folder_to_char[k] for k in sorted(folder_to_char.keys())]
print(f"Total classes loaded: {len(characters)}")

# Function to preprocess a single character image
def preprocess_image(image):
    img_resized = cv2.resize(image, (32, 32))  # Resize to match input size
    img_normalized = img_resized / 255.0       # Normalize to [0, 1]
    return img_normalized

# Function to segment the word image into character images
def segment_word_image(word_img):
    os.makedirs("debug_chars", exist_ok=True)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("debug_thresh.png", binary)

    # Try stronger dilation to separate characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    cv2.imwrite("debug_dilated.png", dilated)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours and sort left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    char_images = []

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        char = binary[y:y+h, x:x+w]

        # Make square image with padding
        size = max(w, h)
        square = 255 * np.ones((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = 255 - char

        # Save and append
        cv2.imwrite(f"debug_chars/char_{i}.png", square)
        char_images.append(square)

    print(f"Total characters detected: {len(char_images)}")
    return char_images


# Predict the Hindi word from an image
def predict_hindi_word(word_img):
    char_images = segment_word_image(word_img)
    predicted_chars = []

    for i, char_img in enumerate(char_images):
        char_img = preprocess_image(char_img)
        char_img = char_img.reshape(1, 32, 32, 1)
        prediction = model.predict(char_img)
        predicted_index = np.argmax(prediction)

        if predicted_index < len(characters):
            predicted_char = characters[predicted_index]
            predicted_chars.append(predicted_char)
            print(f"Character {i+1}: {predicted_char}")
        else:
            print(f"Character {i+1}: Invalid index {predicted_index}, skipping.")

    return ''.join(predicted_chars)

# Run the prediction with a test image
if __name__ == "__main__":
    img_path = 'test_images/rakh.jpg' 

    try:
        input_img = cv2.imread(img_path)
        if input_img is None:
            raise ValueError(f"Failed to load image from path: {img_path}")

        predicted_word = predict_hindi_word(input_img)
        print(f"\nPredicted Hindi Word: {predicted_word}")

    except Exception as e:
        print(f"Error: {e}") 