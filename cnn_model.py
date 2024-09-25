import numpy as np
from keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt
import cv2

# Load the dataset from the saved .npz file
data = np.load('asl_dataset.npz')

# Extract the training, validation, and test datasets
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']
class_names = data['class_names']

print("Dataset loaded successfully!")
print(f"Training set size: {train_images.shape}")
print(f"Validation set size: {val_images.shape}")
print(f"Test set size: {test_images.shape}")

# Normalize the pixel values
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Convert labels to categorical (one-hot encoding)
num_classes = len(class_names)
train_labels_categorical = to_categorical(train_labels, num_classes)
val_labels_categorical = to_categorical(val_labels, num_classes)
test_labels_categorical = to_categorical(test_labels, num_classes)

# Define the image size
image_size = train_images.shape[1]

# Define the CNN model
model = Sequential([
    Input(shape=(image_size, image_size, 1)),  # Assuming grayscale images
    Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same', 
           kernel_initializer='he_normal', bias_initializer='zeros'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Dropout(0.2),
    Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', 
           kernel_initializer='he_normal', bias_initializer='zeros'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax', name='Softmax',
          kernel_initializer='glorot_uniform', bias_initializer='zeros')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(train_images, train_labels_categorical, epochs=20,  validation_data=(val_images, val_labels_categorical), batch_size=32)

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
capture_and_predict(model, class_names, image_size)