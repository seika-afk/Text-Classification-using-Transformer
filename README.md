# Text Classification with Transformer (Keras)

This project demonstrates how to build a Transformer-based model using Keras for binary  classification on the IMDb dataset.
## Overview

The notebook walks through the following steps:

- Tokenize text and convert to padded sequences
- Build a custom embedding layer with token and positional embeddings
- Construct a simple Transformer block from scratch (Multi-head Attention + Feedforward)
- Add  dense layers for classification
- Compile and train the model


## Key Components

### Dataset

- **IMDb Reviews**: The dataset used is built into Keras and contains 25,000 labeled movie reviews for training and 25,000 for testing.

### Tokenization and Padding

- Uses Keras `Tokenizer` and `pad_sequences` to convert raw text into numerical input.

### Custom Layers

- **TokenAndPositionEmbedding**: A custom Keras layer that combines word embeddings and positional embeddings.
- **TransformerBlock**: A Keras layer implementing multi-head self-attention followed by a feed-forward neural network, along with residual connections and layer normalization.

### Model Architecture

- Input → Token & Positional Embedding → Transformer Block → GlobalAveragePooling → Dense → Dropout → Output Dense (Softmax)

### Training

- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy
- Epochs: 2

## How to Run

1. Install TensorFlow 2.x and dependencies:
   ```bash
   pip install tensorflow
   ```

2. Open the notebook in Jupyter or Colab.

3. Run all cells to train and evaluate the model.

## Credits

- Built using TensorFlow and Keras , Using Documentation of Keras.
- Inspired by the official Keras examples and transformer architecture from "Attention is All You Need"
