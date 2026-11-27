# Build an LLM from Scratch

A hands-on notebook project for learning how raw text is prepared for a large language model (LLM). The current notebook follows the text-data workflow from Chapter 2 of *Build a Large Language Model (From Scratch)*.

YouTube: https://www.youtube.com/playlist?list=PLBwE3Q2KZ4kM

A large language model (LLM) is an artificial intelligence model trained on massive amounts of text to understand, generate, and work with human language.

## Project Status

The project currently focuses on the preprocessing foundation of an LLM:

- Loading a short story from `data/the-verdict.txt`
- Splitting text into tokens with regular expressions
- Building a token-to-ID vocabulary
- Encoding and decoding with a custom tokenizer
- Handling unknown words with `<|unk|>` and `<|endoftext|>` tokens
- Using GPT-2 byte-pair encoding through `tiktoken`
- Creating next-token training examples with a sliding window
- Building a PyTorch `Dataset` and `DataLoader`
- Creating token embeddings
- Adding positional embeddings

This repository does not yet implement or train a complete language model.

## Repository Structure

```text
build-llm/
├── ch02.ipynb          # Chapter 2 text-preprocessing experiments
├── data/
│   └── the-verdict.txt # Sample corpus used by the notebook
├── requirements.txt    # Python dependencies
└── README.md
```

## Requirements

- Python 3.11 or a compatible Python 3 version
- A virtual environment is recommended

## Setup

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/poloarol/build-llm.git
   cd build-llm
   ```

2. Create and activate a virtual environment:

   **Windows**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start JupyterLab:

   ```bash
   jupyter lab
   ```

5. Open `ch02.ipynb` and run the cells in order.

## Notebook Walkthrough

The notebook is organized into the following concepts:

### 1. Text loading and tokenization

The sample text is loaded from `data/the-verdict.txt`. If the file is unavailable, the notebook downloads a copy from the source project. Regular expressions are then used to split the text into words and punctuation.

### 2. Vocabulary and token IDs

A vocabulary maps every unique token in the sample corpus to an integer ID. Two tokenizer implementations show:

- A basic vocabulary that raises an error for unseen tokens
- A more robust version that maps unseen tokens to `<|unk|>`

One `KeyError` in the notebook is intentional: it demonstrates the limitation of a fixed vocabulary before the improved tokenizer is introduced.

### 3. GPT-2 byte-pair encoding

`tiktoken` provides the GPT-2 byte-pair encoding tokenizer, which can represent arbitrary text without the same out-of-vocabulary limitation as the simple tokenizer.

### 4. Sliding-window training data

The corpus is encoded into token IDs and divided into overlapping input/target pairs. For every input token sequence, the target sequence is shifted by one token:

```text
input:  [I, had, always, thought]
target: [had, always, thought, Jack]
```

`GPTDatasetV1` and `create_dataloader_v1` package these examples into PyTorch-ready batches.

### 5. Embeddings

PyTorch embedding layers convert token IDs into dense vectors. Learned positional embeddings are then added to token embeddings so the model input contains both token identity and position information.

## Expected Data

The repository includes a local copy of Edith Wharton's short story **"The Verdict"** at `data/the-verdict.txt`. It is used for the educational tokenization and data-loading examples.

## Main Dependencies

- `torch` for tensors, embeddings, datasets, and data loading
- `tiktoken` for GPT-2 byte-pair encoding
- `jupyterlab` for running the notebook
- `numpy`, `pandas`, `matplotlib`, and `tqdm` for later experiments

See `requirements.txt` for the complete list and platform-specific constraints.

## Next Steps

Planned extensions for the project include:

- Implementing self-attention and multi-head attention
- Building transformer blocks
- Assembling a GPT-style model
- Training on a text corpus
- Adding token generation and inference

## Acknowledgements

This notebook is based on the educational structure and Chapter 2 examples from [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.
