
import cv2
import os
import sys

from utils.preprocessing import preprocess_image
from utils.segmentation import segment_digits, draw_bounding_boxes


def main():

    # Step 1: Check image path
    if len(sys.argv) < 2:
        print("Usage: python test_segmentation.py image_path")
        return

    image_path = sys.argv[1]

    # Step 2: Load original image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    # Step 3: Create output directory
    output_folder = "segmentation_output"
    os.makedirs(output_folder, exist_ok=True)

    # Step 4: Preprocess image
    binary = preprocess_image(image)

    cv2.imwrite(
        os.path.join(output_folder, "binary.png"),
        binary
    )

    # Step 5: Segment individual digits
    digits, boxes = segment_digits(binary)

    # Step 6: Draw bounding boxes
    result = draw_bounding_boxes(image, boxes)

    cv2.imwrite(
        os.path.join(output_folder, "bounding_boxes.png"),
        result
    )

    # Step 7: Save individual digit images
    for index, digit in enumerate(digits):

        digit_image = (
            digit.squeeze() * 255
        ).astype("uint8")

        digit_path = os.path.join(
            output_folder,
            f"digit_{index + 1}.png"
        )

        cv2.imwrite(digit_path, digit_image)

    # Step 8: Generate enlarged preview of second digit
    if len(digits) >= 2:

        digit_2_path = os.path.join(
            output_folder,
            "digit_2.png"
        )

        digit_2 = cv2.imread(
            digit_2_path,
            cv2.IMREAD_GRAYSCALE
        )

        if digit_2 is not None:

            enlarged = cv2.resize(
                digit_2,
                (280, 280),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imwrite(
                os.path.join(
                    output_folder,
                    "digit_2_enlarged.png"
                ),
                enlarged
            )

            print("Enlarged second digit saved successfully!")

    # Step 9: Display results
    print("\n========== SEGMENTATION TEST RESULTS ==========")

    print("Detected digits:", len(digits))

    print("Bounding boxes:", boxes)

    print("Individual digit images saved successfully!")

    print("\nOutput folder:", output_folder)

    if len(digits) == 0:
        print("No digits detected. Try another image.")

    print("===============================================")


if __name__ == "__main__":
    main()
