import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import cv2 

# Load the data
train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')

# Extract labels and pixels
labels = train_data['label'].values
pixels = train_data.drop('label', axis=1).values

# Define the image size
image_size = 28

# Reshape the pixels array
pixels = pixels.reshape(-1, image_size, image_size, 1)

# Normalize the pixel values
pixels = pixels / 255.0

# Define the categories excluding J and Z
categories = [chr(i) for i in range(ord('A'), ord('Z') + 1) if i not in [ord('J'), ord('Z')]]

# Create a mapping from original label values to new indices
original_labels = [i for i in range(26) if i not in [9, 25]]  # 9 corresponds to J, 25 corresponds to Z
original_to_new_label_mapping = {original: new for new, original in enumerate(original_labels)}

# Remap the labels to the new range
remapped_labels = np.array([original_to_new_label_mapping[label] for label in labels])

# Convert labels to categorical (one-hot encoding)
num_classes = len(categories)
labels_categorical = to_categorical(remapped_labels, num_classes)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(pixels, labels_categorical, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test), batch_size=32)

# Function to preprocess the image
def preprocess_image(image, image_size, interpolation=cv2.INTER_CUBIC):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to the same size as the training images
    resized = cv2.resize(gray, (image_size, image_size), interpolation=interpolation)
    # Normalize the pixel values
    normalized = resized / 255.0
    # Reshape to match the input shape of the model
    reshaped = normalized.reshape(1, image_size, image_size, 1)
    return reshaped, resized

# Function to capture an image using OpenCV and predict its label
def capture_and_predict(model, categories, image_size):
    # Capture an image from the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to capture an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Wait for the user to press 's' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Preprocess the captured image
            preprocessed_image, resized_image = preprocess_image(frame, image_size, interpolation=cv2.INTER_CUBIC)
            # Predict the label
            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction, axis=1)[0]
            predicted_category = categories[predicted_label]

            # Display the original and preprocessed images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(resized_image, cmap='gray')
            plt.title(f"Preprocessed Image\nPredicted: {predicted_category}")
            plt.axis('off')

            plt.show()

            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Capture an image and predict its label
capture_and_predict(model, categories, image_size)