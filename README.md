# Emotion Detection with CNN

This project implements an emotion detection system using a webcam and a Convolutional Neural Network (CNN) for classification. It utilizes the FER-2013 dataset for training and OpenCV for real-time face and emotion detection. The detected emotions are classified into seven categories and displayed in real-time. You can capture an image and save it in the appropriate folder based on the predicted emotion.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Project Structure

The repository is organized as follows:

```
repo/
│
├── src/
│   └── main.py                     # Main script for running the emotion detection
│
├── utils/
│   └── haarcascade_frontalface_default.xml  # Haar cascade file for face detection
│
├── dataset/                        # Contains the FER-2013 dataset (training and test images)
│   ├── train/                       # Training images organized by emotions
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   └── neutral/
│   └── test/                        # Test images organized by emotions
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── sad/
│       ├── surprise/
│       └── neutral/
│
└── emotions/                        # Folder to save captured images, organized by detected emotions
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## Installation

Follow these steps to set up the environment and run the project.

### Prerequisites
- **Python 3.x**
- **OpenCV** (for face detection and real-time video processing)
- **TensorFlow** (for building and training the CNN model)
- **NumPy** (for data manipulation)

### Install Dependencies

To install the required Python libraries, run the following command in the root of your project directory:

```bash
pip install -r requirements.txt
```

You can manually install the libraries if `requirements.txt` is not provided:

```bash
pip install opencv-python tensorflow numpy
```

## Usage

After successfully setting up the environment, you can run the project using the following command:

1. Make sure you are in the `src/` directory:
   ```bash
   cd src
   ```

2. Run the main script:

   ```bash
   python main.py
   ```

### Instructions:

- The program will open your webcam and start detecting your face.
- The detected emotion will be displayed on the screen in real-time.
- **Press the spacebar** to capture an image. The image will be saved in the corresponding emotion folder inside the `emotions/` directory.
- **Press 'q'** to quit the application.
