import os
import cv2
import numpy as np
import tensorflow as tf
from tf.keras import layers, models 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(label)
    return images, labels

real_images, real_labels = load_images_from_folder('data/real', 0)
fake_images, fake_labels = load_images_from_folder('data/fake', 1)

X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)
X = X / 255.0  # Normalize images

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/test_model.h5')

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()
