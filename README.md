# RNN with Attention Language Model

This repository contains code for training a Recurrent Neural Network (RNN) with an attention mechanism to generate text. The model is designed to learn from the poetry dataset provided by abobster/pushkin_new. The model is implemented using PyTorch.

## Model Architecture

### Long Short-Term Memory (LSTM)

The core of the model is a Long Short-Term Memory (LSTM) layer, a type of recurrent neural network designed to capture long-term dependencies. It consists of a cell state, input gate, forget gate, and output gate.
![LSTM](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Mb_L_slY9rjMr8-IADHvwg.png)

#### The LSTM unit processes input sequences as follows:

1. **Input Processing**: Input features are multiplied by weights in the input gate to determine which information to store in the cell state.

2. **Forget Gate**: The forget gate decides which information from the previous cell state should be discarded.

3. **Cell State Update**: The cell state is updated by combining the information from the input gate and the retained information from the previous cell state.

4. **Output Gate**: The output gate decides what information from the cell state should be the output of the LSTM unit.

### Attention Mechanism

The attention mechanism is integrated into the model using the scaled dot-product attention formula:
$$[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V ]$$
where:
- Q represents the query matrix,
- K represents the key matrix,
- V represents the value matrix,
- sqrt(d_k) is a scaling factor

<img src="https://sun9-71.userapi.com/impf/ZW8EEUxswgqtzxIfY8nQH5M7uOKpeJJT0L3dKw/9AMxCWN2_b4.jpg?size=860x1113&quality=96&sign=d9f40c56ed9773267c7f129149c08a0a&type=album" width="50%" height="50$">

### Training

The model is trained using the AdamW optimizer with a learning rate of `learning_rate`. Training is performed for a maximum of `max_iters` iterations, and the loss is evaluated on both the training and validation sets at regular intervals (`eval_interval`).

### Batch Generation

Training data is divided into batches of size `batch_size`. For each batch, a random starting index is chosen, and subsequences of length `block_size` are extracted for both input and target sequences.

### Generation of New Text

The trained model can generate new text by providing a seed sequence (`start_tokens`). The `generate` method samples new tokens from the model's output logits, allowing for creative and diverse text generation.

## Model Parameters

- `n_embd`: Number of hidden units in the LSTM and attention mechanism.
- `dropout`: Dropout rate applied to the LSTM output.
- `batch_size`: Size of training batches.
- `block_size`: Length of subsequences used for training.
- `learning_rate`: Learning rate for the AdamW optimizer.

## Usage

1. **Dataset Loading**: Load the dataset using the `load_dataset` function from the `datasets` library. Save the training text to 'input.txt'.

2. **Model Initialization**: Create an instance of the `RNNAttentionLanguageModel` class, and print its parameters to the console.

3. **Training Loop**: Train the model using the specified hyperparameters. Training loss and validation loss are printed at regular intervals.

4. **Text Generation**: Use the trained model to generate new text by providing a seed sequence.

## Dependencies
- Python 3.7 or later
- PyTorch
- Hugging Face `datasets` library

## Acknowledgments

The model architecture and training loop are inspired by the works on recurrent neural networks and attention mechanisms in natural language processing. Special thanks to the authors of abobster/pushkin_new dataset.
