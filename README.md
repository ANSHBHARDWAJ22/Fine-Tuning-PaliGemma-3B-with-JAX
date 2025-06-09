# Fine-Tuning-PaliGemma-3B-with-JAX

This repository provides a complete guide for fine-tuning the PaliGemma-3B model using JAX. The setup focuses on efficient fine-tuning of large multimodal models on limited hardware, such as consumer GPUs (T4, A10G) or TPUs.

PaliGemma is a vision-language transformer (VLT) built on Google's Gemma language model architecture. It is designed for tasks where the model generates text outputs conditioned on both an input image and an optional text prefix. Typical use-cases include image captioning, visual question answering (VQA), and multimodal reasoning.

This guide demonstrates how to fine-tune the model efficiently by freezing most parameters and training only the attention layers within the language model.

Contents
Import Dependencies

Download & Configure Model

Shard & Optimize Parameters

Input Preparation

Dataset Iterators

Data Visualization

Training & Evaluation

Concepts and Strategies

1. Import Dependencies
We use a combination of general-purpose libraries and project-specific modules:

Core compute: jax, jax.numpy, numpy

Dataset and file handling: tensorflow, json, os

Tokenization: sentencepiece

Visualization: PIL, IPython.display

Progress indicators: tqdm

Project code: big_vision.models.proj.paligemma, big_vision.trainers.proj.paligemma

Note: TensorFlow is used for input pipeline utilities but disabled for device execution to allow JAX to fully utilize the available hardware.

2. Download and Configure Model
Checkpoints
Model Weights: paligemma-3b-pt-224.f16.npz

Tokenizer: paligemma_tokenizer.model

Both are loaded using kagglehub or manual download.

Configuration
We define a FrozenConfigDict specifying architecture details like layer sizes, number of layers, etc.

Instantiate the PaliGemma model.

Load pretrained weights into the model.

Setup the decode() function for text generation.

The model checkpoint uses float16 precision for memory efficiency.

3. Shard and Optimize Parameters
Fine-tuning large language-vision models on constrained hardware requires careful memory management. Key strategies include:

Freeze Parameters: Only attention layers within the LLM are trainable (llm/layers/attn/). Freezing embeddings, MLPs, and other layers drastically reduces memory consumption.

Mixed Precision: Frozen parameters are cast to float16 to save space; trainable parameters remain in float32 for numerical stability during updates.

Sharding: Parameters are distributed across available devices (GPUs/TPUs) to optimize compute-memory trade-offs.

Sequential Loading: Parameters are loaded incrementally to avoid memory spikes during initialization.

4. Input Preparation
Image Processing
Resize images to (224, 224) resolution.

Normalize pixel values to the range [-1, 1].

Text Tokenization
Tokenize both the text prefix (prompt) and target (suffix) using the SentencePiece tokenizer.

Generate attention masks:

Prefix: Fully attended (all tokens visible to each other)

Suffix: Causal mask (each token can only see preceding tokens)

Output Decoding
The generated token IDs are decoded back into human-readable text after prediction.

5. Dataset Iterators
The training dataset is expected in JSONL format, where each line corresponds to one data sample, structured as:

json
Copy
Edit
{
  "image": "<image filename>",
  "prefix": "<input prompt>",
  "suffix": "<target output text>"
}
Two types of iterators:

Training Iterator: Infinite looping iterator with random shuffling.

Validation Iterator: Single complete pass without shuffling for evaluation consistency.

Each sample yields the image tensor, tokenized prefix, tokenized suffix, and attention masks.

6. Data Visualization
For sanity checks during dataset preparation, we include utilities to:

Render images inline (using base64 HTML)

Display corresponding text captions

This provides a quick preview of the dataset before initiating training.

7. Training and Evaluation
Training Loop
For each training step, compute the loss over the suffix tokens only. Loss excludes prefix tokens and any padding.

Use Stochastic Gradient Descent (SGD) to update only the trainable attention parameters.

Apply learning rate schedules if required for better convergence.

Evaluation Loop
Generate predictions for the validation set using the decode() function.

Since the dataset size might not evenly divide into batches, padding is applied to incomplete batches.

Masks are used to ignore padded predictions when computing evaluation metrics.

8. Concepts and Strategies
This approach relies on several key strategies for scalable fine-tuning:

Frozen Parameters: By freezing most of the model, memory consumption and training time are drastically reduced.

Sharding: Spreads parameters across multiple devices, enabling larger models to fit into limited memory.

Mixed Precision Training: Uses float16 where possible, without compromising numerical stability.

Masked Language Modeling: The model learns to predict only the suffix tokens, conditioned on the image and prefix.

Multi-modal Input: Trains the model to fuse visual and textual context effectively for coherent generation.

Next Steps
Customize the training loop for your dataset.

Experiment with different learning rates and optimizer settings.

Use decode() for generating predictions on new, unseen data.

Evaluate model performance qualitatively and quantitatively.

For advanced applications, consider implementing:

Learning rate warmup and decay schedules

Evaluation on standardized VQA benchmarks

Dataset augmentation techniques for improved generalization

