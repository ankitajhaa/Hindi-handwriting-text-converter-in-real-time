import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle

# Load preprocessed data
X_train = np.load('data\X_train_devanagari.npy')
y_train = np.load('data\y_train_devanagari.npy')
X_test = np.load('data\X_test_devanagari.npy')
y_test = np.load('data\y_test_devanagari.npy')

# Ensure correct input shape
input_shape = (32, 32, 1)  # 32x32 grayscale images

model = models.Sequential([
    layers.Input(shape=input_shape),  # (32, 32, 1)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),

    layers.Flatten(),  # Flatten the feature map
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stop])

# Ensure model directory exists before saving
os.makedirs("models", exist_ok=True)

# Save the model
model.save('models/devanagari_cnn_model.h5')

# Save the folder_to_char mapping using pickle
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

with open('label_map.pkl', 'wb') as f:
    pickle.dump(folder_to_char, f)

print("Training completed and model saved!")
print("Label map saved to 'label_map.pkl'")