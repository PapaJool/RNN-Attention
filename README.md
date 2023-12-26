# RNN Attention Language Model

This repository contains the implementation of an RNN attention-based language model trained on Pushkin's texts. The model is designed to generate text character by character, and it uses an attention mechanism to capture context dependencies within the input sequence.

## Model Overview

The language model is built using PyTorch and consists of the following components:

- **RNN (Recurrent Neural Network):** A recurrent layer processes the input sequence, capturing sequential dependencies in the data.
  
- **Attention Mechanism:** An attention mechanism allows the model to focus on different parts of the input sequence during processing. This helps the model learn long-range dependencies.

- **Softmax Layer:** The final layer applies softmax to the output, generating a probability distribution over the vocabulary for each time step.

## Training

The model is trained using the provided training script (`train.py`). During training, the model minimizes the cross-entropy loss between predicted and actual characters. Training is performed using the AdamW optimizer, and the process can be monitored for both training and validation losses.

To train the model, follow these steps:

1. Install dependencies: `pip install -r requirements.txt`
2. Run the training script: `python code/train.py`

## Text Generation

After training, the model can be used to generate new text using the provided generation script (`generate.py`). The user can specify the temperature parameter, controlling the randomness of the generated text. Higher temperatures lead to more diverse output, while lower temperatures make the output more deterministic.

To generate text, run the following:

```bash
python code/generate.py --temperature 0.8 --max_tokens 500
Adjust the temperature and max_tokens as needed.

Project Structure

code: Contains Python scripts for training and generating text.
data: Stores input data, such as the text dataset.
checkpoints: Can store pre-trained model weights or other important model components.
docs: Documentation folder (optional).
README.md: Overview of the project and instructions for usage.
requirements.txt: List of dependencies.
.gitignore: Specifies files and directories to be ignored by Git.
Acknowledgments

The model was trained on the Pushkin dataset from Hugging Face Datasets.
