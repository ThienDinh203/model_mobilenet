import cv2
import numpy as np
import tensorflow.lite as tflite
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # Expected input shape

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

# Your class labels (in order used during training)
class_labels = ["Awake", "Yawning", "Sleepy"]  # example labels
last_label = "Start..."
last_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    current_time = time.time()

    if current_time - prev_time >= 1 and len(faces) > 0:
        prev_time = current_time

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))  # Resize to model's input size
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            face = np.expand_dims(face, axis=0).astype(np.float32) / 255.0  # Normalize

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], face)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (3,)

            # Get class with highest probability
            max_index = np.argmax(predictions)
            max_label = class_labels[max_index]
            confidence = predictions[max_index]

            last_label = f"{max_label} ({confidence:.2f})"
            last_color = (0, 255, 0) if max_label == "Awake" else (0, 165, 255) if max_label == "Yawning" else (0, 0, 255)

    # Draw bounding box & text
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), last_color, 2)
        cv2.putText(frame, last_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
