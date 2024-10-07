# ASL Hand Gesture Recognition using MediaPipe and CNN

This project is a pipeline for recognizing American Sign Language (ASL) gestures. The steps include creating a dataset, preprocessing the images, training a Convolutional Neural Network (CNN) model, detecting hand signs, and evaluating the model's accuracy.

## Project Overview
![ASL Recognition Demo](gifs/demo.gif)

## Project Structure

- `create_dataset.py`: A script for capturing images of hand gestures for each letter in the ASL alphabet.
- `preprocessing.py`: A script for preprocessing the dataset (resizing, grayscale conversion, thresholding, etc.) and splitting it into training, validation, and test sets.
- `cnn_model.py`: A script for training the CNN model on the preprocessed dataset.
- `sign_detector.py`: A script for real-time sign detection using the trained model.
- `evaluation.py`: A script to evaluate the accuracy of the trained model using test data.

---

## 1. Prerequisites

Before running the scripts, install the required libraries by running:

```bash
pip install opencv-python mediapipe matplotlib numpy scikit-learn tensorflow
```

## 2. Dataset Creation

To create a dataset of hand images for ASL letters, run the `create_dataset.py` script. The script uses your webcam to capture hand gestures for the following letters: **A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y**. Each gesture is captured using MediaPipe's hand tracking system, and the captured images are saved in a structured folder system.

### Run:

```bash
python create_dataset.py
```

### Key Instructions:

- You will be prompted to press any key to start capturing images for each letter.
- The captured images will be saved in `asl_dataset/<letter>/` for each corresponding letter.
- To quit image capturing for a letter, press **'q'**.

---

## 3. Preprocessing and Splitting the Dataset

After creating the dataset, the images need to be preprocessed and split into training, validation, and test sets. Run the `preprocessing.py` script to achieve this.

### Preprocessing steps:

- Convert images to grayscale
- Apply Gaussian blur
- Perform thresholding to enhance contrast
- Resize images to 32x32 pixels
- Normalize pixel values to the range [0, 1]

### Run:

```bash
python preprocessing.py
```

The preprocessed images and corresponding labels will be saved in a compressed `.npz` file called `asl_dataset.npz`. The dataset is automatically split into training, validation, and test sets.

---

## 4. Training the CNN Model

Once the dataset is ready, you can train the CNN model using the `cnn_model.py` script. This script loads the preprocessed dataset, builds a CNN, and trains the model.

### Run:

```bash
python train_model.py
```

The script will output the training progress, including loss and accuracy for both the training and validation sets. After training, the model is saved as `asl_cnn_model.h5`.

---

## 5. Sign Detection

After training the model, you can use the `sign_detector.py` script for real-time ASL gesture detection using your webcam. The script captures your hand gestures and uses the trained model to predict the corresponding ASL alphabet.

### Run:

```bash
python sign_detector.py
```

### Key Instructions:

- The script will open your webcam and start detecting hand gestures.
- The detected gesture will be displayed on the screen in real-time.
- Press **'q'** to quit the program.

---

## 6. Model Evaluation

If you wish to evaluate the accuracy of the model on the test dataset, run the `evaluation.py` script. This script loads the preprocessed test dataset and evaluates the model's accuracy.

### Run:

```bash
python evaluation.py
```

The script will ask the user to capture a 5 second video of each hand gesture. It will then output the accuracy, precision, recall and F-1 score of the evaluation. Press **'c'** to capture each letter and press **'q'** to quit.

---

## Summary of Steps:

1. **Create Dataset**: Use the webcam to capture ASL hand gestures (`create_dataset.py`).
2. **Preprocess Dataset**: Preprocess and split the dataset (`preprocessing.py`).
3. **Train Model**: Train a CNN model on the preprocessed data (`cnn_model.py`).
4. **Sign Detector**: Use the trained model to detect ASL gestures in real-time (`sign_detector.py`).
5. **Evaluate Model**: Evaluate the model's performance on test data (`evaluation.py`).

---

## Authors

- Muhammad Zarif Bin Rozaini
- Muhamad Syamil Imran Bin Mohd Mansor
- Muhammad Amir Faris Bin Ahsan Nudin
- Amgad Elrashid Gurashid Eltayeb
- Balchi Maher M. N.
