import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the generator model
generator = load_model('models/conditional/generator.h5')

# Function to generate images
def generate_images(generator, num_images, latent_dim, labels):
    latent_points = np.random.randn(num_images, latent_dim)
    labels = np.array(labels).reshape(-1, 1)
    generated_images = generator.predict([latent_points, labels])
    return generated_images

# Create directories for generated images
output_dir = 'generated_images'
os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'pneumonia'), exist_ok=True)

# Generate and save images
num_images_per_class = 100  # Number of images to generate per class
latent_dim = 100  # Latent dimension size

# Generate images for normal class
normal_labels = [0] * num_images_per_class
generated_normal_images = generate_images(generator, num_images_per_class, latent_dim, normal_labels)
for i, image in enumerate(generated_normal_images):
    filename = os.path.join(output_dir, 'normal', f'image_{i}.png')
    tf.keras.preprocessing.image.save_img(filename, image)

# Generate images for pneumonia class
pneumonia_labels = [1] * num_images_per_class
generated_pneumonia_images = generate_images(generator, num_images_per_class, latent_dim, pneumonia_labels)
for i, image in enumerate(generated_pneumonia_images):
    filename = os.path.join(output_dir, 'pneumonia', f'image_{i}.png')
    tf.keras.preprocessing.image.save_img(filename, image)

print("Generated images saved successfully.")
