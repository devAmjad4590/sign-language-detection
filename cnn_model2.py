import numpy as np
from keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

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
model.fit(train_images, train_labels_categorical, epochs=40, validation_data=(val_images, val_labels_categorical), batch_size=32)

# Function to preprocess the image
def preprocess_image(image, image_size, interpolation=cv2.INTER_AREA):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to the same size as the training images
    resized = cv2.resize(gray, (image_size, image_size), interpolation=interpolation)
    resized = resized.reshape(1, image_size, image_size, 1)
    resized = resized / 255.0
    

    # Normalize the pixel values
    # Reshape to match the input shape of the model
    return resized

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to detect hand and predict the ASL letter
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

        # Flip the frame to avoid mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(frame_rgb)

        # Check if any hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Expand the bounding box a bit
                box_margin = 20
                x_min = max(x_min - box_margin, 0)
                x_max = min(x_max + box_margin, w)
                y_min = max(y_min - box_margin, 0)
                y_max = min(y_max + box_margin, h)

                # Draw bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Crop the hand region
                hand_region = frame[y_min:y_max, x_min:x_max]

                # Preprocess the cropped hand image
                preprocessed_image = preprocess_image(hand_region, image_size)

                # Predict the label for the cropped hand image
                prediction = model.predict(preprocessed_image)
                predicted_label = np.argmax(prediction, axis=1)[0]
                predicted_category = categories[predicted_label]

                # Display the prediction on the frame
                cv2.putText(frame, f"Predicted: {predicted_category}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame with hand detection and prediction
        cv2.imshow('Hand Detection and Prediction', frame)

        # Wait for the user to press 's' to stop capturing
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
# Capture an image and predict its label
capture_and_predict(model, class_names, image_size)