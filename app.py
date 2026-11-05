import streamlit as st
import cv2
import numpy as np
import pandas as pd

from predict import load_cnn_model, predict_number
from utils.segmentation import draw_bounding_boxes


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Handwritten Number Recognition",
    page_icon="🔢",
    layout="wide"
)


# -----------------------------------
# LOAD CNN MODEL
# -----------------------------------

@st.cache_resource
def get_model():
    return load_cnn_model()


# -----------------------------------
# APPLICATION TITLE
# -----------------------------------

st.title("🔢 Multi-Digit Handwritten Number Recognition")

st.write(
    "Upload an image containing handwritten digits. "
    "Our CNN model will recognize the individual digits "
    "and combine them into a complete number."
)

st.divider()


# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your handwritten number image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Unable to read the uploaded image.")

    else:

        # Display original image
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Uploaded Image")
        st.image(
            image_rgb,
            caption="Original handwritten image",
            width=500
        )

        # -----------------------------------
        # PREDICTION BUTTON
        # -----------------------------------

        if st.button(
            "Recognize Number",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Processing image and predicting digits..."
                ):

                    # Load trained CNN model
                    model = get_model()

                    # Predict complete number
                    number, digit_results, boxes, binary = (
                        predict_number(image, model)
                    )

                    # Draw bounding boxes
                    boxed_image = draw_bounding_boxes(
                        image,
                        boxes
                    )

                    boxed_image_rgb = cv2.cvtColor(
                        boxed_image,
                        cv2.COLOR_BGR2RGB
                    )

                # -----------------------------------
                # DISPLAY FINAL PREDICTION
                # -----------------------------------

                st.success("Recognition completed!")

                st.subheader("Predicted Number")

                st.markdown(
                    f"<h1 style='text-align:center;"
                    f"font-size:65px;'>{number}</h1>",
                    unsafe_allow_html=True
                )

                st.write(
                    f"**Total digits detected:** {len(boxes)}"
                )

                st.divider()

                # -----------------------------------
                # DISPLAY PROCESSING RESULTS
                # -----------------------------------

                st.subheader("Image Processing Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Binary Image**")
                    st.image(
                        binary,
                        caption="Preprocessed image",
                        width="stretch"
                    )

                with col2:
                    st.write("**Digit Segmentation**")
                    st.image(
                        boxed_image_rgb,
                        caption="Detected digit regions",
                        width="stretch"
                    )

                st.divider()

                # -----------------------------------
                # INDIVIDUAL DIGIT PREDICTIONS
                # -----------------------------------

                st.subheader("Individual Digit Predictions")

                results_df = pd.DataFrame(digit_results)

                results_df["confidence"] = (
                    results_df["confidence"] * 100
                ).round(2)

                results_df.rename(
                    columns={
                        "position": "Position",
                        "prediction": "Predicted Digit",
                        "confidence": "Confidence (%)"
                    },
                    inplace=True
                )

                st.dataframe(
                    results_df,
                    hide_index=True,
                    use_container_width=True
                )

                # -----------------------------------
                # CONFIDENCE SCORES
                # -----------------------------------

                st.subheader("Digit Confidence Scores")

                for result in digit_results:

                    confidence = result["confidence"]

                    st.write(
                        f"Digit {result['position']}: "
                        f"{result['prediction']} "
                        f"({confidence * 100:.2f}%)"
                    )

                    st.progress(confidence)

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )

                st.info(
                    "Check your model path, image format, "
                    "and preprocessing configuration."
                )

else:

    st.info(
        "Please upload a handwritten number image to begin."
    )


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Intelligent Recognition. Powered by AI."
)
