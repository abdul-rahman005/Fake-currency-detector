import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define dataset folders
REAL_FOLDER = 'dataset/real'
FAKE_FOLDER = 'dataset/fake'

# Ensure dataset directories exist
if not os.path.exists(REAL_FOLDER) or not os.path.exists(FAKE_FOLDER):
    raise FileNotFoundError("❌ Dataset folders not found. Ensure 'dataset/real' and 'dataset/fake' exist.")

# Load images and labels
def load_images_and_labels():
    images, labels = [], []
    real_count, fake_count = 0, 0  # Debugging counters

    for filename in os.listdir(REAL_FOLDER):
        img_path = os.path.join(REAL_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(1)  # ✅ Real Note
        real_count += 1

    for filename in os.listdir(FAKE_FOLDER):
        img_path = os.path.join(FAKE_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(0)  # ✅ Fake Note
        fake_count += 1

    print(f"🔍 Total Real Notes: {real_count}")
    print(f"🔍 Total Fake Notes: {fake_count}")

    if real_count == 0 or fake_count == 0:
        raise ValueError("❌ Dataset is imbalanced! Add more images.")

    images = np.array(images).astype(np.float32) / 255.0  # Normalize
    labels = np.array(labels).reshape(-1, 1)  # Correct shape
    return images, labels

# Create improved CNN Model
def create_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', 
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Load and split data
    images, labels = load_images_and_labels()
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and train model
    model = create_model()
    
    # Train with validation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=30,
        steps_per_epoch=len(X_train) // 32,
        validation_steps=len(X_val) // 32
    )

    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"\n✅ Validation Accuracy: {val_accuracy*100:.2f}%")
    
    # Save the model
    os.makedirs('model', exist_ok=True)
    model.save('model/fake_note_detector.h5')
    print("✅ Model training complete and saved to model/fake_note_detector.h5")
