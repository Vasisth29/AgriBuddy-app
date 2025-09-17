# FILE: train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os
import json

# --- CONFIGURATION ---
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'
model_save_path = 'models/soil_model.h5'
class_indices_path = 'models/class_indices.json'
img_width, img_height = 128, 128
batch_size = 32
epochs = 25  # Increased epochs for better learning

# --- DATA AUGMENTATION ---
# Create more training data by applying random transformations to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Test data should not be augmented, only rescaled for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# --- DATA GENERATORS ---
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# --- BUILD A MORE POWERFUL CNN MODEL ---
model = Sequential([
    # Layer 1: Find basic features
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Layer 2: Find more complex features
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Layer 3: Find even more complex features
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Flatten the features and prepare for classification
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Dropout layer to prevent overfitting
    
    # Final output layer: one neuron per soil type
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- TRAIN THE MODEL ---
print("Starting model training...")
model.fit(train_generator, validation_data=test_generator, epochs=epochs)

# --- SAVE THE FINAL MODEL AND CLASS INDICES ---
os.makedirs('models', exist_ok=True)
model.save(model_save_path)
with open(class_indices_path, 'w') as f:
    json.dump(train_generator.class_indices, f)

print(f"\nModel training complete!")
print(f"Model saved at {model_save_path}")
print(f"Class indices saved at {class_indices_path}")