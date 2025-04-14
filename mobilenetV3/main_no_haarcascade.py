import cv2
import numpy as np
import tensorflow.lite as tflite
import time
import os

# Disable GPU for TensorFlow Lite
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Open webcam
cap = cv2.VideoCapture(0)

# Your class labels
class_labels = ["Awake", "Yawning", "Sleepy"]
prev_time = time.time()
last_label = "Start..."
last_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - prev_time >= 1:  # Run every 1 second
        prev_time = current_time

        # Preprocess full frame
        input_img = cv2.resize(frame, (224, 224))  # Resize to model input size
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        max_index = np.argmax(predictions)
        max_label = class_labels[max_index]
        confidence = predictions[max_index]

        last_label = f"{max_label} ({confidence:.2f})"
        last_color = (0, 255, 0) if max_label == "Awake" else (0, 165, 255) if max_label == "Yawning" else (0, 0, 255)

    # Display label
    cv2.putText(frame, last_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
