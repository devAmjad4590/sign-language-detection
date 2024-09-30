import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# Define the size to which the hand image will be resized for the CNN model
image_size = 32

# To keep track of the previously predicted letter
previous_letter = None

# Load the trained ASL recognition model
model = load_model('asl_cnn_model.h5')

# Initialize MediaPipe Hand Detector with configurations for real-time detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define the function to preprocess the hand image before sending it to the CNN model
def preprocess_image(image):
    # Convert the hand image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flip the image horizontally to match the user's perspective
    gray = cv2.flip(gray, 1)
    
    # Apply Gaussian blur to smooth the image
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply binary thresholding to create a black and white image
    _, thresholded = cv2.threshold(gray, 161, 255, cv2.THRESH_BINARY)
    
    # Resize the image to the model's required input size (32x32 pixels)
    resized = cv2.resize(thresholded, (image_size, image_size))
    
    # Normalize the pixel values from 0-255 to 0-1 for the CNN
    normalized = resized / 255.0
    
    # Reshape the image to match the model's input shape (1, 32, 32, 1)
    reshaped = np.reshape(normalized, (1, image_size, image_size, 1))
    
    return reshaped, thresholded

# Define a function to map the CNN model's prediction to the corresponding ASL letter
def predict_asl_letter(prediction):
    asl_letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'  # Exclude 'J' and 'Z' due to motion in the signs
    return asl_letters[prediction]

# Start video capture for live ASL recognition
cap = cv2.VideoCapture(0)

# Initialize variables to store recognized text and time information
recognized_text = ""
asl_letter = ""
last_letter_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the video frame to RGB as MediaPipe requires this color format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands using MediaPipe
    result = hands.process(frame_rgb)

    # If hands are detected in the frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the dimensions of the video frame
            h, w, c = frame.shape

            # Initialize variables for the bounding box around the hand
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Loop through hand landmarks to find the min/max coordinates for the bounding box
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add a margin to the bounding box for better cropping
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop the region of the hand from the frame
            hand_image = frame[y_min:y_max, x_min:x_max]

            # Preprocess the cropped hand image
            preprocessed_image, resized_image = preprocess_image(hand_image)

            # Use the trained model to predict the ASL letter
            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Calculate the confidence of the prediction

            # Map the predicted label to the corresponding ASL letter
            asl_letter = predict_asl_letter(predicted_label)
            
            # Draw a bounding box around the detected hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display the predicted ASL letter and confidence on the video frame
            cv2.putText(frame, f'ASL Letter: {asl_letter} ({confidence:.2f}%)', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show the preprocessed hand image in a separate window
            cv2.imshow('Preprocessed Image', resized_image)

    # Display the video frame with hand detection and ASL letter recognition
    cv2.imshow('ASL Recognition', frame)

    # Create a blank white image to display the recognized text
    text_window = np.ones((200, 500, 3), dtype=np.uint8) * 255  # White background

    # Display the recognized text in the new window
    cv2.putText(text_window, 'Recognized Text:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(text_window, recognized_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the recognized text window
    cv2.imshow('Recognized Text', text_window)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        break

    current_time = time.time()

    # Track how long the current ASL letter stays the same
    if asl_letter != previous_letter:  # Check if the letter has changed
        last_letter_time = time.time()  # Reset the timer when the letter changes
        previous_letter = asl_letter  # Set the current letter as the previous one
    else:
        # If the letter hasn't changed, append it to recognized text after 2.5 seconds
        if current_time - last_letter_time >= 2.5:
            recognized_text += asl_letter  # Add the letter to the text
            last_letter_time = time.time()  # Reset the timer

    asl_letter = ""  # Reset the letter after it's added to the text

    # If 'c' is pressed, clear the recognized text
    if key == ord('c'):
        recognized_text = ''

    # If 's' is pressed, add a space to the recognized text
    if key == ord('s'):
        recognized_text += " "

    if key == ord('b'):
        recognized_text = recognized_text[:-1]  # Remove the last character (backspace)


# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
