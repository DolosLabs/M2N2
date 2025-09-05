# M2N2: Competition and Attraction Improve Model Fusion (PyTorch Implementation)

This repository contains a PyTorch Lightning reproduction of the paper **"Competition and Attraction Improve Model Fusion"** ([arXiv:2508.16204](https://arxiv.org/abs/2508.16204)) by João P. Abrantes, Robert Tjarko Lange, and Yujin Tang.

The code replicates **Experiment 1:** evolving MNIST classifiers entirely *from scratch* using the proposed evolutionary algorithm **Model Merging of Natural Niches (M2N2)**.

---

## Table of Contents

- [Introduction](#introduction)  
- [Core Concepts of M2N2](#core-concepts-of-m2n2)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Expected Output](#expected-output)  
- [Code Structure](#code-structure)  
- [Citation](#citation)  
- [License](#license)

---

## Introduction

Model merging combines multiple specialized machine learning models into a single, more capable one. Traditional methods often rely on fixed layer boundaries and hand-designed merge rules. **M2N2** introduces an **evolutionary approach** that automates this process, demonstrating that models can be evolved from scratch without gradient-based training.

---

## Core Concepts of M2N2

1. **Evolving Merging Boundaries**  
   Merge models at a random split point in their flattened parameter vectors, allowing a flexible search of parameter combinations. Implemented via `_crossover` using **Spherical Linear Interpolation (SLERP)**.

2. **Diversity via Competition ("Natural Niches")**  
   Fitness is calculated based on relative performance across the population: models are rewarded for correctly classifying examples that others fail on. Implemented in `_update_fitness`.

3. **Attraction-Based Mating**  
   Parents are selected intelligently: the first by overall fitness, the second by an "attraction score" that complements the first. Implemented in `_select_parents`.

---

## Requirements

- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [PyTorch Lightning](https://www.pytorchlightning.ai/)  
- torchvision  
- numpy  
- tqdm  

Install dependencies:

```bash
pip install torch torchvision pytorch-lightning numpy tqdm
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/M2N2-pytorch.git
cd M2N2-pytorch
```

---

## Usage

Run the MNIST evolution experiment:

```bash
python m2n2_mnist.py
```

The script will:

1. Download MNIST automatically.  
2. Initialize a random population of small MLP classifiers.  
3. Run the M2N2 evolutionary loop.  
4. Display progress and final test accuracy.

---

## Expected Output

Example progress bar during training:

```
Evolving Models: 100%|██████████| 50000/50000 [15:30<00:00, 53.72it/s, best_fitness=5012.34, test_acc=92.45%]
Evolution finished.
Final Test Accuracy of the best model: 92.45%
```

> Note: Results will vary based on hyperparameters and random seeds.

---

## Code Structure

- **Helper functions:** `get_params`, `set_params`, `slerp`  
- **Model Architecture:** `MLP` for MNIST  
- **Data Module:** `MNISTDataModule` using PyTorch Lightning  
- **M2N2 Class:** Handles archive, fitness calculation, selection, crossover, mutation, and evolution loop  
- **Main Execution Block:** Configures and runs the experiment

---

## Citation

```bibtex
@article{abrantes2025competition,
  title={Competition and Attraction Improve Model Fusion},
  author={Abrantes, Jo{\~a}o P and Lange, Robert Tjarko and Tang, Yujin},
  journal={arXiv preprint arXiv:2508.16204},
  year={2025}
}
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

