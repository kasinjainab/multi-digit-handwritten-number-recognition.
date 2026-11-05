
import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Configuration
EPOCHS = 10
BATCH_SIZE = 128
MODEL_PATH = "models/cnn_model.keras"

# 2. Create models directory
os.makedirs("models", exist_ok=True)

# 3. Load MNIST dataset
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 4. Normalize images
# Convert pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 5. Reshape images for CNN
# Original: (28, 28)
# New: (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# 6. Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])

# 7. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8. Display architecture
model.summary()

# 9. Train model
print("\nStarting CNN training...")

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# 10. Evaluate model
print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# 11. Save model
model.save(MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")
