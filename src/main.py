import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Define paths
train_path = '../dataset/train/'  # Path to training images (FER-2013)
emotions_path = '../emotions'      # Path to save the classified images

# Prepare emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize a counter for saved images per emotion
emotion_counters = {emotion: 0 for emotion in emotion_labels}

# Step 1: Load the FER-2013 dataset and prepare data for training KNN
def load_data():
    data = []
    labels = []
    for emotion in emotion_labels:
        folder = os.path.join(train_path, emotion)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Check if the image was loaded properly
                img_resized = cv2.resize(img, (48, 48))  # Resize to 48x48 as in FER-2013
                data.append(img_resized.flatten())  # Flatten the image to 1D array
                labels.append(emotion_labels.index(emotion))
    return np.array(data), np.array(labels)

# Load data
X, y = load_data()

# Step 2: Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Step 3: Video capture and face detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../utils/haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()
    
    # Resize the frame for better performance
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))  # Resize face to match FER-2013
        face_flattened = face_resized.flatten().reshape(1, -1)

        # Step 4: Predict the emotion using KNN
        emotion_index = knn.predict(face_flattened)[0]
        emotion = emotion_labels[emotion_index]

        # Draw rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with the detected face and emotion
    cv2.imshow('Emotion Detection', frame)

    # Check for key presses
    key = cv2.waitKey(10)
    
    # Press 'q' to exit
    if key == ord('q'):
        break
    # Press spacebar to save the image
    elif key == 32:  # Spacebar key
        if len(faces) > 0:  # Only save if a face is detected
            emotion_counters[emotion] += 1  # Increment the counter for the detected emotion
            save_folder = os.path.join(emotions_path, emotion)
            os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist
            file_name = f'{emotion}-{emotion_counters[emotion]}.png'  # Save as emotion-1, emotion-2, etc.
            cv2.imwrite(os.path.join(save_folder, file_name), frame)
            print(f"Picture saved in {save_folder}/{file_name}")

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
