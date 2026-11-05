import os
import cv2
import numpy as np
import tensorflow as tf

from utils.preprocessing import preprocess_image
from utils.segmentation import segment_digits


# Path to the trained CNN model
MODEL_PATH = "models/cnn_model.keras"


def load_cnn_model():
    """
    Load the trained CNN model.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    print("CNN model loaded successfully!")

    return model


def predict_number(image, model):
    """
    Predict a handwritten multi-digit number.

    Parameters:
        image: Original input image.
        model: Trained CNN model.

    Returns:
        number: Predicted complete number.
        digit_results: Individual digit predictions and confidences.
        boxes: Bounding boxes of detected digits.
        binary: Preprocessed binary image.
    """

    if image is None:
        raise ValueError("Invalid input image.")

    # Step 1: Preprocess the original image
    binary = preprocess_image(image)

    # Step 2: Segment individual digits
    segmented_digits, boxes = segment_digits(binary)

    if len(segmented_digits) == 0:
        raise ValueError(
            "No digits detected. Please upload another image."
        )

    # Step 3: Create a batch for CNN prediction
    digit_batch = np.stack(segmented_digits, axis=0)

    # Step 4: Predict all digits
    probabilities = model.predict(
        digit_batch,
        verbose=0
    )

    # Step 5: Get predicted classes
    predictions = np.argmax(probabilities, axis=1)

    # Step 6: Get confidence scores
    confidences = np.max(probabilities, axis=1)

    # Step 7: Combine individual predictions
    number = "".join(
        str(int(digit)) for digit in predictions
    )

    # Step 8: Store individual prediction details
    digit_results = []

    for index, (digit, confidence) in enumerate(
        zip(predictions, confidences)
    ):
        digit_results.append({
            "position": index + 1,
            "prediction": int(digit),
            "confidence": float(confidence)
        })

    return number, digit_results, boxes, binary


if __name__ == "__main__":

    # Load trained model
    model = load_cnn_model()

    # Change this to your test image path
    image_path = "sample.png"

    # Read original image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Predict complete number
    number, digit_results, boxes, binary = predict_number(
        image,
        model
    )

    # Display final prediction
    print("\n========== PREDICTION RESULTS ==========")

    print("Predicted Number:", number)

    print("\nIndividual Digit Predictions:")

    for result in digit_results:
        print(
            f"Digit {result['position']}: "
            f"{result['prediction']} | "
            f"Confidence: {result['confidence'] * 100:.2f}%"
        )

    print("\nTotal Digits Detected:", len(boxes))

    print("========================================")
