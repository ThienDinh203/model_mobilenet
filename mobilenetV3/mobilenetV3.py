import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# If you want to force CPU (optional):
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ======= Config =======
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 45
dataset_path = "../face_dataset_224_augment"  # change to your path

# ======= Load Dataset =======
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"  # for 3+ classes
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ======= Prefetch (optional for speed) =======
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ======= Build Model =======
base_model = MobileNetV3Large(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze feature extractor

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = Dense(128)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = Dense(64)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)  # 3 classes

model = Model(inputs=base_model.input, outputs=output)

# ======= Compile Model =======
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ======= Summary =======
model.summary()

# ======= Train =======
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ======= Save and Convert to TFLite =======
# model.save("mobilenetv3_large_model")

# # Convert to TFLite
# converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv3_large_model")
# tflite_model = converter.convert()

# # Save the TFLite model
# with open("mobilenetv3_large.tflite", "wb") as f:
#     f.write(tflite_model)

# print("✅ TFLite model saved as mobilenetv3_large.tflite")

saved_model_path = "drowsiness_detector.keras"
model.save(saved_model_path)

# Convert to TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_saved_model(model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the .tflite file
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

model.summary()  # Check if model is properly built

print(f"✅ Model successfully converted and saved as {saved_model_path}")
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
