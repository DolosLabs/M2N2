# M2N2: "Competition and Attraction Improve Model Fusion"
# A PyTorch Lightning Implementation for Evolving MNIST Classifiers from Scratch
# Source: arXiv:2508.16204

# === 1. Imports ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from copy import deepcopy

# === 2. Helper Functions ===

def get_params(model):
    """Flattens all parameters of a model into a single 1D tensor."""
    return torch.cat([p.data.flatten() for p in model.parameters()])

def set_params(model, params_vector):
    """Loads a flat parameter vector into a model."""
    offset = 0
    for p in model.parameters():
        p.data.copy_(params_vector[offset:offset + p.numel()].view(p.size()))
        offset += p.numel()

def slerp(p1, p2, t, dot_threshold=0.9995):
    """
    Performs Spherical Linear Interpolation (SLERP) between two tensors.
    [cite_start]This is used for the crossover operation in the evolutionary algorithm. [cite: 3]
    """
    dot = torch.sum(p1 * p2) / (torch.norm(p1) * torch.norm(p2))
    omega = torch.acos(dot.clamp(-1, 1))
    
    if dot > dot_threshold:
        # If vectors are very close, use linear interpolation for stability
        return (1 - t) * p1 + t * p2

    term1 = torch.sin((1 - t) * omega) / torch.sin(omega) * p1
    term2 = torch.sin(t * omega) / torch.sin(omega) * p2
    return term1 + term2

# === 3. Model Architecture ===

class MLP(nn.Module):
    """
    A simple two-layer feedforward neural network for MNIST classification.
    [cite_start]This matches the model architecture described in the paper's experiments. [cite: 4]
    """
    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === 4. LightningDataModule ===

class MNISTDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling the MNIST dataset.
    This encapsulates data loading, transformations, and splitting.
    """
    def __init__(self, data_dir: str = "./", batch_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self):
        # Download data if it doesn't exist
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            # We use the full training set for evaluating fitness, as required by the
            # [cite_start]resource competition mechanism. [cite: 3]
            self.train_dataset = MNIST(self.data_dir, train=True, transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = MNIST(self.data_dir, train=False, transform=self.transform)
            
    def train_dataloader(self):
        # Dataloader for the training set (used for fitness evaluation)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        # Dataloader for the test set (used for final performance reporting)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        
# === 5. The M2N2 Evolutionary Algorithm ===

class M2N2:
    """
    Implements the core logic of the Model Merging of Natural Niches (M2N2) algorithm.
    This class manages the archive of models, the evolutionary loop, parent selection,
    crossover, and fitness evaluation.
    """
    
    def __init__(self, model_class, train_loader, test_loader, archive_size=20, device='cuda'):
        self.model_class = model_class
        self.test_loader = test_loader
        self.archive_size = archive_size
        self.device = device
        self.epsilon = 1e-8 # Small constant for numerical stability

        # Cache the full training dataset on the target device for faster evaluation
        print("Caching training data on device...")
        self.train_images = torch.cat([batch[0] for batch in train_loader], dim=0).to(device)
        self.train_labels = torch.cat([batch[1] for batch in train_loader], dim=0).to(device)
        
        self.num_train_samples = len(self.train_labels)
        self.forward_passes = 0
        
        # --- Initialize the archive with randomly initialized models ---
        print(f"Initializing archive with {archive_size} random models...")
        self.archive = [self.model_class().to(device) for _ in range(archive_size)]
        self.scores = torch.zeros(self.archive_size, self.num_train_samples, device=device)
        self.fitness = torch.zeros(self.archive_size, device=device)
        self.best_model = None

    @torch.no_grad()
    def _evaluate_model(self, model):
        """Evaluates a single model on the cached training data and returns per-sample scores."""
        preds = model(self.train_images).argmax(dim=1)
        scores = (preds == self.train_labels).float()
        self.forward_passes += 1
        return scores
        
    def _update_fitness(self):
        """
        Updates the fitness of all models in the archive based on competition.
        [cite_start]This implements the "Natural Niches" concept from Equation 3 in the paper. [cite: 3]
        """
        # (2) Diversity via Competition: Calculate sum of scores per data point (z_j)
        z_j = self.scores.sum(dim=0) + self.epsilon
        # For binary scoring, the capacity (c_j) is 1. Fitness is the sum of shared scores.
        self.fitness = (self.scores / z_j).sum(dim=1)
        
        # Sort the archive by fitness in descending order
        sorted_indices = torch.argsort(self.fitness, descending=True)
        self.archive = [self.archive[i] for i in sorted_indices]
        self.scores = self.scores[sorted_indices]
        self.fitness = self.fitness[sorted_indices]

    def _select_parents(self):
        """
        Selects two parent models for crossover.
        Parent A is chosen based on fitness. Parent B is chosen based on its
        [cite_start]"attraction" score relative to Parent A (Equation 4). [cite: 3, 4]
        """
        # Select Parent A: Higher fitness leads to higher selection probability
        fitness_probs = self.fitness / self.fitness.sum()
        parent_a_idx = np.random.choice(self.archive_size, p=fitness_probs.cpu().numpy())
        parent_a = self.archive[parent_a_idx]
        
        # (3) Attraction-based Mating: Calculate attraction scores for all potential partners
        z_j = self.scores.sum(dim=0) + self.epsilon
        parent_a_scores = self.scores[parent_a_idx]
        
        # Complementarity is high if model B succeeds where model A fails
        complementarity = F.relu(self.scores - parent_a_scores) # max(s_b - s_a, 0)
        
        attraction_scores = (complementarity / z_j).sum(dim=1)
        attraction_scores[parent_a_idx] = -1 # Ensure a model cannot be its own partner
        
        # Select Parent B based on attraction probability
        if attraction_scores.sum() <= 0:
            # Fallback: if no model is attractive, choose a random different partner
            parent_b_idx = np.random.choice([i for i in range(self.archive_size) if i != parent_a_idx])
        else:
            attraction_probs = F.softmax(attraction_scores, dim=0)
            parent_b_idx = np.random.choice(self.archive_size, p=attraction_probs.cpu().numpy())
            
        parent_b = self.archive[parent_b_idx]
        return parent_a, parent_b

    def _crossover(self, parent_a, parent_b):
        """
        Merges two parent models using SLERP with a random split-point to create a child.
        [cite_start]This implements the "Evolving Merging Boundaries" concept. [cite: 1, 3]
        """
        # (1) Evolving Merging Boundaries
        p1 = get_params(parent_a)
        p2 = get_params(parent_b)
        
        num_params = len(p1)
        split_point = np.random.randint(1, num_params) # Randomly choose where to split params
        mix_ratio = np.random.rand() # Randomly choose the interpolation ratio
        
        # Merge the two parts of the parameter vectors separately
        child_params_part1 = slerp(p1[:split_point], p2[:split_point], mix_ratio)
        child_params_part2 = slerp(p1[split_point:], p2[split_point:], 1 - mix_ratio)
        
        child_params = torch.cat([child_params_part1, child_params_part2])
        
        child_model = self.model_class().to(self.device)
        set_params(child_model, child_params)
        return child_model

    @torch.no_grad()
    def test_model(self, model):
        """Evaluates a model's accuracy on the unseen test set."""
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total
        
    def evolve(self, num_evaluations=50000):
        """Runs the main evolutionary loop for a fixed number of forward passes."""
        
        # Initial evaluation of the randomly generated archive
        for i in range(self.archive_size):
            self.scores[i] = self._evaluate_model(self.archive[i])

        pbar = tqdm(total=num_evaluations, initial=self.forward_passes, desc="Evolving Models")
        
        while self.forward_passes < num_evaluations:
            self._update_fitness()
            
            # Select parents and create a child via crossover
            parent_a, parent_b = self._select_parents()
            child_model = self._crossover(parent_a, parent_b)
            
            # Evaluate the new child
            child_scores = self._evaluate_model(child_model)
            
            # [cite_start]The child replaces the worst individual in the archive if its fitness is higher. [cite: 3]
            worst_fitness = self.fitness[-1]
            
            # Calculate child's potential fitness in the current ecosystem
            current_z_j = self.scores.sum(dim=0) + self.epsilon
            child_fitness = (child_scores / current_z_j).sum()

            if child_fitness > worst_fitness:
                self.archive[-1] = child_model
                self.scores[-1] = child_scores
            
            # Log progress periodically
            if self.forward_passes % 1000 == 0:
                self.best_model = deepcopy(self.archive[0])
                test_acc = self.test_model(self.best_model)
                pbar.set_postfix({"best_fitness": f"{self.fitness[0]:.2f}", "test_acc": f"{test_acc:.2f}%"})
            
            pbar.update(1)
        
        pbar.close()
        print("\nEvolution finished.")
        # Final evaluation of the best model found
        self.best_model = deepcopy(self.archive[0])
        final_test_acc = self.test_model(self.best_model)
        print(f"Final Test Accuracy of the best model: {final_test_acc:.2f}%")

# === 6. Main Execution Block ===

if __name__ == '__main__':
    # --- Configuration ---
    # [cite_start]These parameters match the "Evolution from Scratch" experiment. [cite: 4]
    ARCHIVE_SIZE = 20
    BATCH_SIZE = 1024 # Larger batch size for faster evaluation, doesn't affect algorithm
    NUM_EVALUATIONS = 50000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    pl.seed_everything(42)

    # 1. Setup Data using the LightningDataModule
    mnist_dm = MNISTDataModule(batch_size=BATCH_SIZE)
    mnist_dm.prepare_data()
    mnist_dm.setup('fit')
    mnist_dm.setup('test')
    
    # 2. Initialize the M2N2 Algorithm Orchestrator
    m2n2_evolver = M2N2(
        model_class=MLP,
        train_loader=mnist_dm.train_dataloader(),
        test_loader=mnist_dm.test_dataloader(),
        archive_size=ARCHIVE_SIZE,
        device=DEVICE
    )
    
    # 3. Start the Evolutionary Process
    m2n2_evolver.evolve(num_evaluations=NUM_EVALUATIONS)
