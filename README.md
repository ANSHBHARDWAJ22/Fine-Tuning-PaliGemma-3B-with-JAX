# Fine-Tuning PaliGemma-3B with JAX

This repository provides a complete guide for fine-tuning the **PaliGemma-3B** model using **JAX**. The setup focuses on **efficient fine-tuning of large multimodal models on limited hardware**, such as consumer GPUs (T4, A10G) or TPUs.

PaliGemma is a **vision-language transformer (VLT)** built on top of Google's Gemma language model architecture. It is designed for generating text conditioned on both images and optional text prefixes. Example use cases include:

- Image Captioning
- Visual Question Answering (VQA)
- Multimodal Reasoning Tasks

This guide demonstrates how to fine-tune the model efficiently by **freezing most parameters** and **training only the attention layers** of the language model.

---

## Table of Contents

1. [Import Dependencies](#1-import-dependencies)
2. [Download & Configure Model](#2-download--configure-model)
3. [Shard & Optimize Parameters](#3-shard--optimize-parameters)
4. [Input Preparation](#4-input-preparation)
5. [Dataset Iterators](#5-dataset-iterators)
6. [Data Visualization](#6-data-visualization)
7. [Training & Evaluation](#7-training--evaluation)
8. [Concepts and Strategies](#8-concepts-and-strategies)
9. [Next Steps](#9-next-steps)

---

## 1. Import Dependencies

We use a combination of standard libraries and project-specific modules:

- **Core Compute:** `jax`, `jax.numpy`, `numpy`
- **Data Handling:** `tensorflow`, `json`, `os`
- **Tokenization:** `sentencepiece`
- **Visualization:** `PIL`, `IPython.display`
- **Progress Bars:** `tqdm`
- **Project Modules:** `big_vision.models.proj.paligemma`, `big_vision.trainers.proj.paligemma`

Note: TensorFlow is used for input pipeline utilities but its device execution is disabled to allow JAX full access to available hardware.

---

## 2. Download & Configure Model

### Checkpoints

- **Model Weights:** `paligemma-3b-pt-224.f16.npz`
- **Tokenizer:** `paligemma_tokenizer.model`

Download them using `kagglehub` or manually.

### Configuration

- Define a `FrozenConfigDict` specifying model architecture (layers, heads, embedding dimensions).
- Instantiate the **PaliGemma** model.
- Load the pretrained weights into the model.
- Setup the `decode()` function for generating text from output tokens.

The pretrained model uses **float16 precision** to optimize memory usage.

---

## 3. Shard & Optimize Parameters

Fine-tuning large models on constrained hardware requires efficient memory management. The key strategies used:

- **Freeze Parameters:** Only the **attention layers** (`llm/layers/attn/`) are trainable. Embeddings, MLPs, and other components remain frozen.
- **Mixed Precision:** Frozen parameters are cast to `float16`; trainable parameters remain in `float32` for stability.
- **Sharding:** Parameters are distributed across available devices (multi-GPU/TPU environments).
- **Sequential Loading:** Model parameters are loaded in parts to avoid memory spikes during initialization.

---

## 4. Input Preparation

### Image Preprocessing

- Resize all images to `(224, 224)` resolution.
- Normalize pixel values to the range `[-1, 1]`.

### Text Tokenization

- Tokenize both the **prefix** (input prompt) and the **suffix** (target text) using the provided SentencePiece tokenizer.
- Generate **attention masks**:
  - **Prefix:** Fully attended (all tokens see each other).
  - **Suffix:** Causal (each token only attends to previous tokens).

### Output Decoding

Generated token IDs are decoded back into human-readable text using the tokenizer's decode function.

---

## 5. Dataset Iterators

The training dataset is expected in **JSONL** format. Example:

```json
{
  "image": "<image filename>",
  "prefix": "<input prompt>",
  "suffix": "<target output text>"
}
```
## 6. Data Visualization

For sanity checking the dataset, utilities are included to:

- Render images inline using **base64 HTML encoding**
- Display the corresponding text captions alongside the image

This helps ensure that both **images and text inputs** are correctly aligned before starting training.

---

## 7. Training & Evaluation

### Training Loop

- Compute **loss only on the suffix tokens**.
- Loss **excludes**:
  - Prefix tokens
  - Padding tokens
- Use **Stochastic Gradient Descent (SGD)** to update **only** the attention layers.
- Optionally, implement **learning rate schedules** for better convergence and stability.

### Evaluation Loop

- Use the `decode()` function to generate predictions for the **validation dataset**.
- Apply **padding** to incomplete batches to maintain consistent batch sizes.
- Use **attention masks** to **ignore padded predictions** during evaluation metric computation.

---

## 8. Concepts and Strategies

This fine-tuning approach uses several key strategies to maximize efficiency:

- **Frozen Parameters:** Freezing most of the model **drastically reduces memory usage** and **speeds up training**.
- **Sharding:** Splits model parameters **across multiple devices** (GPUs/TPUs) for handling large models.
- **Mixed Precision Training:** Uses **float16** wherever possible **without compromising numerical stability**.
- **Masked Language Modeling:** The model **predicts only the suffix tokens**, conditioned on the input **image** and **prefix text**.
- **Multimodal Input:** Trains the model to **combine visual and textual features** for coherent and contextually accurate generation.

---

## 9. Next Steps

- **Customize the training loop** for your dataset and task.
- Experiment with different **optimizers** and **learning rate schedules**.
- Use `decode()` for generating predictions on **new, unseen data**.
- **Evaluate** the model using both:
  - **Quantitative** metrics (e.g., BLEU, CIDEr, ROUGE)
  - **Qualitative** inspection (visual + text results)

### For Advanced Applications:

- Implement **learning rate warmup and decay schedules** for better convergence.
- Evaluate on **standard VQA or captioning benchmarks** for robust performance comparisons.
- Apply **dataset augmentation** techniques to improve **generalization**.

---


