# Image Captioning

## Overview

This project implements an image captioning system that generates descriptive captions for images using a Vision Transformer (ViT) as the encoder and GPT-2 as the decoder. The project employs advanced machine learning techniques to train and evaluate the model, leveraging the Flickr8k dataset as the primary data source.

The project is structured to showcase how to utilize pre-trained transformer models for vision and language tasks. The implementation combines computer vision and natural language processing into a unified framework for image captioning.

## Features

- **Transformer-based Architecture**: Combines a ViT encoder for image feature extraction and a GPT-2 decoder for caption generation.
- **Dynamic Data Augmentation**: Enhances the training data using transformations like resizing, random flips, and rotations.
- **Evaluation with ROUGE Metrics**: Measures the similarity between generated captions and ground truth captions.
- **Configurable Hyperparameters**: Easily adjustable settings for batch size, learning rate, epochs, and more.
- **Pre-trained Models**: Utilizes pre-trained Vision Transformer and GPT-2 models to reduce computational costs and training time.
- **Interactive Caption Generation**: Allows dynamic temperature adjustment for experimenting with caption diversity.

## Dataset

The project uses the **Flickr8k dataset**, which consists of 8,000 images, each paired with five descriptive captions. The dataset is divided into training and validation sets using an 80-20 split. Image files and captions are stored in structured directories for easy access.

## Preprocessing

- Images are resized and normalized to match the input requirements of the Vision Transformer.
- Captions are tokenized and padded to a maximum length of 128 tokens.

## Implementation Details

### Model Architecture

The model employs a Vision Transformer (ViT) as the encoder to extract high-dimensional image features. These features are passed to a GPT-2 decoder, which generates natural language captions. Key architectural features include:

- **Encoder**: Pre-trained ViT (`google/vit-base-patch16-224`) processes images into feature embeddings.
- **Decoder**: Pre-trained GPT-2 generates captions using embeddings from the encoder.
- **Vision-Language Fusion**: The encoder-decoder framework seamlessly combines visual and textual data.

### Data Augmentation

To improve generalization and robustness, the training pipeline includes:

- Resizing and normalizing images.
- Random horizontal flips and rotations.
- Color jittering for brightness, contrast, and saturation adjustments.

## Training and Evaluation

### Training

The model is trained using the `Seq2SeqTrainer` from the Transformers library, which automates training, validation, and logging.

### Metrics

- **ROUGE-2** is used to evaluate the similarity between generated and ground truth captions.

### Hyperparameters

- **Batch size**: 16  
- **Learning rate**: 2e-5  
- **Maximum sequence length**: 128  
- **Number of epochs**: 10  
- **Weight decay**: 0.02  

## Results

The model achieves coherent and descriptive captions for images in the Flickr8k dataset.

### Example Captions

**Image 1**:  
- **Generated Caption**: A man riding a bicycle in the park.  
- **True Caption**: A man riding a bike on a path surrounded by trees.  

**Image 2**:  
- **Generated Caption**: A dog jumping to catch a frisbee in the air.  
- **True Caption**: A brown dog leaping into the air to catch a frisbee.  

### Interface Screenshots
![WhatsApp Görsel 2024-11-28 saat 13 11 16_fc0247f9](https://github.com/user-attachments/assets/bcdb8d24-7328-42d6-9d80-42e4b2c1b78c)


![WhatsApp Görsel 2024-11-28 saat 14 32 01_ff25e83f](https://github.com/user-attachments/assets/87c90af2-1e82-401f-94c7-ab5d37cf720c)


