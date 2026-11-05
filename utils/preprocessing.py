
import cv2
import numpy as np


def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Convert to white digits on black background
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    return binary


def prepare_digit_for_cnn(digit):
    h, w = digit.shape

    # Create a square canvas
    size = max(h, w) + 8
    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = digit

    # Resize digit to 20x20
    resized = cv2.resize(
        square,
        (20, 20),
        interpolation=cv2.INTER_AREA
    )

    # Place inside 28x28 MNIST canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized

    # Normalize pixel values
    normalized = canvas.astype("float32") / 255.0

    # Add channel dimension
    normalized = np.expand_dims(normalized, axis=-1)

    return normalized
