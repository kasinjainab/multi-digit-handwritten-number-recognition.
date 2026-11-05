# Multi-Digit Handwritten Number Recognition

An AI-powered application that recognizes handwritten multi-digit numbers from uploaded images using a Convolutional Neural Network (CNN) and OpenCV.

## Overview

This project identifies individual handwritten digits from an image, predicts each digit using a CNN trained on the MNIST dataset, and combines the predictions to produce the complete number.

The application provides a simple interactive interface built with Streamlit.

## Features

- Upload handwritten number images.
- Preprocess images using OpenCV.
- Detect and segment individual digits using contour detection.
- Recognize digits using a CNN model.
- Display the predicted multi-digit number.
- Show individual digit predictions and confidence scores.
- Visualize image preprocessing and digit segmentation.

## Technologies Used

- **Python** – Core programming language
- **TensorFlow / Keras** – CNN model development
- **OpenCV** – Image preprocessing and digit segmentation
- **NumPy** – Numerical operations
- **Pandas** – Prediction results
- **Streamlit** – Interactive web application

## Project Workflow

1. Upload an image containing handwritten digits.
2. Convert the image into grayscale and apply binary thresholding.
3. Detect individual digits using OpenCV contour detection.
4. Sort detected digits from left to right.
5. Resize and normalize each digit to 28 × 28 pixels.
6. Pass the segmented digits to the trained CNN model.
7. Predict individual digits and calculate confidence scores.
8. Combine the predictions to display the complete number.

## Project Structure

```text
multi_digit_recog/
│
├── models/
│   └── cnn_model.keras
│
├── utils/
│   ├── preprocessing.py
│   └── segmentation.py
│
├── segmentation_output/
│
├── app.py
├── predict.py
├── test_segmentation.py
├── sample.png
├── requirements.txt
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

**2. Navigate to the project directory**

```bash
cd multi_digit_recog
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the virtual environment**

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the Streamlit application**

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Model Information

- Dataset: MNIST handwritten digit dataset
- Model: Convolutional Neural Network (CNN)
- Input dimensions: 28 × 28 × 1
- Output classes: 10 (digits 0–9)

## Limitations

- Touching or overlapping handwritten digits may not be segmented correctly.
- Recognition performance can vary depending on handwriting style and image quality.
- The model is trained on MNIST, so handwriting that differs significantly from the training data may affect predictions.

## Future Improvements

- Improve segmentation for touching and overlapping digits.
- Evaluate and improve recognition performance on diverse handwriting samples.
- Explore sequence-based models for direct multi-digit recognition.
- Deploy the application online.
