
import cv2
from utils.preprocessing import prepare_digit_for_cnn


def segment_digits(binary_image):
    """
    Detect individual handwritten digits using contours.

    Input:
        binary_image: White digits on black background

    Output:
        List of segmented digit images
        List of bounding boxes
    """

    # Find external contours
    contours, _ = cv2.findContours(
        binary_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_boxes = []

    # Detect bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Filter small noise
        if area < 15:
            continue

        # Filter extremely small regions
        if h < 5 or w < 2:
            continue

        digit_boxes.append((x, y, w, h))

    # Sort digits from left to right
    digit_boxes.sort(key=lambda box: box[0])

    segmented_digits = []

    for x, y, w, h in digit_boxes:
        # Crop individual digit
        digit = binary_image[y:y+h, x:x+w]

        # Prepare for CNN
        processed_digit = prepare_digit_for_cnn(digit)

        segmented_digits.append(processed_digit)

    return segmented_digits, digit_boxes


def draw_bounding_boxes(image, boxes):
    """
    Draw rectangles around detected digits.
    """

    output = image.copy()

    for index, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            str(index + 1),
            (x, max(y - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return output
