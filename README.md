# GenStrucs

This repository provides the fine-tuning datasets, configuration examples, and auxiliary analysis scripts used in our catalyst–adsorbate structure generation study.

## Code Attribution

The generative model architecture, tokenizer design, and training procedures are based on CatGPT, a Transformer-based framework for catalyst structure generation.

Please refer to the official CatGPT repository for the original implementation:

https://github.com/SeoinBack/CatGPT

## Fine-Tuning Dataset

The fine-tuning datasets are organized by adsorbate species. In this study, four nitrogen-related adsorbates were considered:

- N
- NH
- NNH
- NHNH

Each dataset contains tokenized catalyst–adsorbate structures used for adsorbate-specific fine-tuning. The tokenized representation includes catalyst composition, adsorbate identity, simulation cell information, atom types, and atomic coordinates.

## Overview of the Generation Pipeline

The dataset generation follows a two-stage workflow:

1. **Model training / fine-tuning**  
2. **Large-scale structure generation (inference)**

This repository focuses on **stage (2)**.  
For **model training and fine-tuning**, users should refer to the original CatGPT implementation.

---

## Model Training and Fine-Tuning

The generative model architecture, tokenizer design, and training procedures are based on **CatGPT**, a Transformer-based framework for catalyst structure generation.

> **Important**  
> All training codes are **not reimplemented here**.  
> Please refer to the official CatGPT repository for full details:

---

## Structure Generation (Inference)

Once a trained or fine-tuned model checkpoint is available, catalyst structures are generated using the provided generation script.

### Generation Script

Structure generation is performed using the `script/generate_gpu.py`

The output is a serialized .pkl file containing generated token sequences.
At this stage, no decoding, filtering, or geometry relaxation is applied.

## Decoding and Conversion to XYZ Format

The generated .pkl files are converted into readable atomistic structures using: `script/transfer_xyz.py` 

This script decodes token sequences and writes valid structures into simple XYZ files.


## Data Availability

The fine-tuning datasets used in this work are provided in this repository. Additional large-scale generated `.pkl` files and corresponding `.xyz` structure files are available at:
https://www.scidb.cn/s/6FvUNf
