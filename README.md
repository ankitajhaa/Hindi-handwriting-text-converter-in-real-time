# 📝 Hindi Handwriting Text Converter (Real-Time)

A machine learning project that segments and recognizes **handwritten Hindi characters** from word images in real time using a trained **Convolutional Neural Network (CNN)**.

---

## 🚀 Features

- 🧠 Trained on 36 Devanagari characters
- 🔍 Real-time character segmentation from word images
- 🧾 Accurate Hindi word prediction using deep learning
- 📸 Easy testing with custom handwritten images

---

## 🧪 How It Works

1. **Image Preprocessing**
   - Converts word image to grayscale and binary
   - Segments characters using contour detection

2. **Character Prediction**
   - Each character is passed through a trained CNN model
   - Labels are mapped using a pre-saved `label_map.pkl`

3. **Final Output**
   - Characters are stitched together to form the final Hindi word

---

## 📂 Folder Structure

Hindi-Handwriting-Text-Converter/
├── main.py 
├── preprocess.py 
├── train.py 
├── data/ 
├── dataset/
├── models/ 
├── test_images/
├── debug_chars/ 
├── label_map.pkl 
├── README.md 
└── .gitignore
