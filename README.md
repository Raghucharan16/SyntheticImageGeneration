# Synthetic Image Generation for Chest X-ray Dataset

This project aims to generate synthetic chest X-ray images using conditional Generative Adversarial Networks (cGANs) and further improve the accuracy of pneumonia detection using these generated images.

## Overview

Chest X-ray images are crucial for diagnosing various respiratory diseases, including pneumonia. However, obtaining a large and diverse dataset of chest X-ray images for training machine learning models can be challenging due to privacy concerns and data availability. This project addresses this challenge by generating synthetic chest X-ray images using cGANs, which can then be used to augment existing datasets.

## Features

- **Synthetic Image Generation**: Utilizes conditional Generative Adversarial Networks (GANs) to generate synthetic chest X-ray images.
- **Data Augmentation**: Augments existing chest X-ray datasets with synthetic images to improve model performance.
- **Pneumonia Detection**: Trains a deep learning model for pneumonia detection using the augmented dataset.
- **Web Application**: Provides a Streamlit-based web application for users to upload chest X-ray images and receive predictions on pneumonia diagnosis.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Raghucharan16/SyntheticImageGeneration.git

   cd SyntheticImageGeneration
   pip install -r requirements.txt

## Usage

- Generate Synthetic Images:

Run the GAN model to generate synthetic chest X-ray images.
- Data Augmentation:

Augment existing chest X-ray datasets with the generated synthetic images.
Train Pneumonia Detection Model:

- Train a deep learning model for pneumonia detection using the augmented dataset.
- Run the Web Application:

Start the Streamlit web application to allow users to upload chest X-ray images for pneumonia diagnosis.
## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

