# Stochastic Representation Aggregation for ViT

This repository provides a framework for evaluating a new aggregation strategy for Vision Transformers (ViT) representations, called **stochastic representation aggregation**. The framework supports linear probing experiments across various pretraining methods and fine-grained classification datasets using Hugging Face models and datasets.

## Features
- Aggregation strategies: CLS token, global average pooling, stochastic aggregation
- Linear probing with frozen backbone
- Fine-grained classification datasets (CIFAR-100, DTD, FGVC Aircraft, Fungi, etc.)
- GPU-only training

## Usage

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare config
Edit `configs/config.yaml` to specify models, datasets, and training parameters.

### 3. Run experiments
```bash
python main.py --model <model_name> --dataset <dataset_name> --strategy <strategy> --config config.yaml --checkpoint_root_path results --checkpoint
```

### 4. Aggregation strategies
- `cls`: Use CLS token
- `avg`: Global average pooling
- `stochastic`: Sampled representation (mean + noise * sqrt(variance))

## File Structure
- `main.py`: Entry point, handles config and experiment loop
- `trainer.py`: Training and evaluation logic
- `model_loader.py`: Loads models and preprocessors
- `dataset_loader.py`: Loads datasets and builds dataloaders
- `aggregation.py`: Aggregation functions
- `configs/config.yaml`: Experiment configuration

## Citation
If you use this codebase, please cite the original author.

---
For questions or issues, please open an issue on GitHub.
