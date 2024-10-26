import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_path = '../dataset/train/'
emotions_path = '../emotions'

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_counters = {emotion: 0 for emotion in emotion_labels}

def load_data():
    data = []
    labels = []
    for emotion in emotion_labels:
        folder = os.path.join(train_path, emotion)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (48, 48))
                data.append(img_resized.flatten())
                labels.append(emotion_labels.index(emotion))
    return np.array(data), np.array(labels)

X, y = load_data()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../utils/haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_flattened = face_resized.flatten().reshape(1, -1)
        emotion_index = knn.predict(face_flattened)[0]
        emotion = emotion_labels[emotion_index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == 32 and len(faces) > 0:
        emotion_counters[emotion] += 1
        save_folder = os.path.join(emotions_path, emotion)
        os.makedirs(save_folder, exist_ok=True)
        file_name = f'{emotion}-{emotion_counters[emotion]}.png'
        cv2.imwrite(os.path.join(save_folder, file_name), frame)
        print(f"Picture saved in {save_folder}/{file_name}")

cap.release()
cv2.destroyAllWindows()
