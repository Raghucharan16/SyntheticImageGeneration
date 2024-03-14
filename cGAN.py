import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

# Define constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Define paths to your dataset
TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
TEST_DIR = "chest_xray/test"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# Load the pre-trained BigGAN model from TensorFlow Hub
biggan_module = hub.Module("https://tfhub.dev/deepmind/biggan-128/2")

# Modify the last layer to output two classes
output = Dense(NUM_CLASSES, activation='softmax')(biggan_module.outputs[0])

# Create the fine-tuned BigGAN model
model = Model(inputs=biggan_module.inputs, outputs=output)

# Freeze the layers of the BigGAN model
for layer in model.layers[:-1]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

# Train the model
history = model.fit(train_generator,
                    epochs=NUM_EPOCHS,
                    validation_data=val_generator)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Sample and save generated images
r, c = 2, 5
noise = np.random.normal(0, 1, (r * c, 128))
sampled_labels = np.array([0] * (r * c // 2) + [1] * (r * c // 2))
gen_imgs = model.predict([noise, sampled_labels.reshape(-1, 1)])

# Create directories to save generated images
os.makedirs("generated_images/normal", exist_ok=True)
os.makedirs("generated_images/pneumonia", exist_ok=True)

# Save the generated images
for i in range(r * c):
    # Determine the directory based on the label
    label = "normal" if sampled_labels[i] == 0 else "pneumonia"
    save_path = os.path.join("generated_images", label)
    # Save the generated image
    cv2.imwrite(os.path.join(save_path, "%d_%d.png" % (i // c, i % c)), (gen_imgs[i, :, :, 0] * 255).astype(np.uint8))
