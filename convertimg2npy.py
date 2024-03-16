import os
import numpy as np
from PIL import Image

def load_images_from_dir(directory, label, target_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):  #you should specify which image extension you deal with
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize(target_size)  # Resize the image
            image = np.array(image)
            images.append(image)
            labels.append(label)
    return images, labels

def create_npy_files(data_dir, output_dir):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    val_images = []
    val_labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            if label == 'normal':
                class_label = 0
            elif label == 'pneumonia':
                class_label = 1
            else:
                continue
            
            images, labels = load_images_from_dir(os.path.join(label_dir, 'train'), class_label)
            train_images.extend(images)
            train_labels.extend(labels)
            
            images, labels = load_images_from_dir(os.path.join(label_dir, 'test'), class_label)
            test_images.extend(images)
            test_labels.extend(labels)
            
            images, labels = load_images_from_dir(os.path.join(label_dir, 'val'), class_label)
            val_images.extend(images)
            val_labels.extend(labels)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    np.save(os.path.join(output_dir, 'val_images.npy'), val_images)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)

data_dir = 'chest_xray'  # path to your dataset directory
output_dir = 'generatedimgs'  # path to the directory where you want to save .npy files
create_npy_files(data_dir, output_dir)
