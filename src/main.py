import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model_path = '../models/emotion_cnn_model.h5'
train_path = '../dataset/train/'
emotions_path = '../emotions'
face_cascade_path = '../utils/haarcascade_frontalface_default.xml'

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_counters = {emotion: 0 for emotion in emotion_labels}

use_pretrained = input("Do you want to use the pretrained model? (yes/no): ").strip().lower()

if use_pretrained == 'yes' and os.path.exists(model_path):
    cnn_model = tf.keras.models.load_model(model_path)
    print(f"Loaded pretrained model from {model_path}")
else:
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
                    data.append(img_resized)
                    labels.append(emotion_labels.index(emotion))
        return np.array(data), np.array(labels)

    X, y = load_data()
    X = X.reshape(-1, 48, 48, 1) / 255.0

    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(emotion_labels), activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X, y, epochs=10, batch_size=32)
    
    cnn_model.save(model_path)
    print(f"Model trained and saved to {model_path}")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        emotion_index = np.argmax(cnn_model.predict(face_resized))
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