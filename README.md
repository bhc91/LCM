# Mini Large Concept Model (LCM)

This repo contains a lightweight, fully-scripted prototype inspired by Meta AI’s **Large Concept Model**.  
It trains a contrastive image-text model on a 1 k-sample Flickr8k subset (editable) and supports an optional audio branch, a small concept memory and interactive retrieval demos.

## Features
* **Dual encoders** – ResNet-50 for images, BERT-base for text (*Wav2Vec 2.0* for audio; audio branch is defined but not trained in the quick run).
* **Shared 512-d embedding space** with InfoNCE contrastive loss.
* **Learnable concept memory** (100 keys) that adds a concept-alignment loss.
* **Caption augmentation & pseudo object labels** to enrich concept grounding.
* **Dataset** – Flickr8k subset (1 000 images × 5 captions) for lightning-fast experiments.
* **Quick-test toggles** – edit `MAX_TRAIN_STEPS` / `MAX_EVAL_STEPS` to switch between a smoke test and a full epoch.
* **Interactive retrieval** – call `find_similar_images()` to show the top-*k* images for any text query.

---

## Quick Start (Notebook Version)

1. **Open** `lcm_mini.ipynb` in Jupyter or Google Colab.

2. **Install deps** – run the first cell  
   ```bash
   pip install torch torchvision transformers datasets tqdm matplotlib pillow
3. **Train & plot** – just hit Run all.
The notebook trains on 10 mini-batches (~20 min on a Colab T4) and shows two plots.
