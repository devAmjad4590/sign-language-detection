import cv2
import os
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# List of letters to capture
letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# Create a directory to store the dataset if it doesn't exist
dataset_dir = 'asl_dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Function to capture images for each letter
def capture_images_for_letter(letter, num_images=200):
    # Create a directory for the letter if it doesn't exist
    letter_dir = os.path.join(dataset_dir, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Press any key to start capturing images for letter: {letter}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame to avoid mirror effect
        frame = cv2.flip(frame, 1)

        # Display the instruction label on the frame
        cv2.putText(frame, "Press any key to start capturing images", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Hand Detection', frame)

        # Break the loop if any key is pressed
        if cv2.waitKey(1) & 0xFF != 255:
            break

    print(f"Capturing images for letter: {letter}")
    count = 0
    while count < num_images:
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
                box_margin = 30
                x_min = max(x_min - box_margin, 0)
                x_max = min(x_max + box_margin, w)
                y_min = max(y_min - box_margin, 0)
                y_max = min(y_max + box_margin, h)

                # Draw bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Crop the hand region
                hand_region = frame[y_min:y_max, x_min:x_max]

                # Save the cropped hand image
                img_path = os.path.join(letter_dir, f'{letter}_{count}.jpg')
                cv2.imwrite(img_path, hand_region)
                count += 1

        # Display the letter label on the frame
        cv2.putText(frame, f"Letter: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Hand Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Loop through each letter and capture images
for letter in letters:
    capture_images_for_letter(letter)

print("Dataset creation complete!")