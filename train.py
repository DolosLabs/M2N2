# m2n2_mnist_evolver.py
"""
Rewritten M2N2 implementation aligned with:
"Abrantes et al., Competition and Attraction Improve Model Fusion" (arXiv:2508.16204v1)

Key additions vs. your original:
 - alpha (competition strength) as in Eq.5
 - capacity c_j handling for binary vs continuous scores
 - Gaussian mutation during reproduction (used when evolving from scratch)
 - numerically robust slerp, device/dtype safety, and selection fallbacks
 - optional caching of the full train set on device (cache_on_device)
"""

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
import os

# -------------------------
# Helper utilities
# -------------------------
def get_params(model, device=None, dtype=None):
    """Return a single 1D tensor with all parameters detached and moved to device/dtype."""
    params = [p.detach().view(-1).to(device=device, dtype=dtype) for p in model.parameters()]
    if len(params) == 0:
        return torch.tensor([], device=device, dtype=dtype)
    return torch.cat(params)

def set_params(model, params_vector):
    """Load a flat parameter vector into a model. Ensures device/dtype match."""
    if not torch.is_tensor(params_vector):
        params_vector = torch.tensor(params_vector)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    params_vector = params_vector.to(device=device, dtype=dtype)

    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(params_vector[offset:offset + n].view(p.size()))
        offset += n
    assert offset == params_vector.numel(), "Parameter-vector size mismatch when loading parameters."

def slerp(p1, p2, t, dot_threshold=0.9995, eps=1e-8):
    """
    Numerically stable SLERP between two 1D parameter tensors (p1,p2).
    Falls back to linear interpolation if norms are tiny or vectors nearly colinear.
    Ensures device/dtype alignment.
    """
    if not torch.is_tensor(p1):
        p1 = torch.tensor(p1)
    if not torch.is_tensor(p2):
        p2 = torch.tensor(p2)

    device = p1.device
    dtype = p1.dtype
    p2 = p2.to(device=device, dtype=dtype)

    norm1 = torch.norm(p1)
    norm2 = torch.norm(p2)
    if norm1 < eps or norm2 < eps:
        t_t = torch.tensor(t, device=device, dtype=dtype)
        return (1 - t_t) * p1 + t_t * p2

    dot = (p1 * p2).sum() / (norm1 * norm2 + eps)
    dot = dot.clamp(-1.0, 1.0)

    if dot.abs() > dot_threshold:
        t_t = torch.tensor(t, device=device, dtype=dtype)
        return (1 - t_t) * p1 + t_t * p2

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < eps:
        t_t = torch.tensor(t, device=device, dtype=dtype)
        return (1 - t_t) * p1 + t_t * p2

    t_t = torch.tensor(t, device=device, dtype=dtype)
    term1 = torch.sin((1 - t_t) * omega) / sin_omega * p1
    term2 = torch.sin(t_t * omega) / sin_omega * p2
    return term1 + term2

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    """Two-layer MLP used in the paper's MNIST experiments (≈19k params)."""
    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------
# DataModule
# -------------------------
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MNIST(self.data_dir, train=True, transform=self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        # deterministic order important when caching
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True)

# -------------------------
# M2N2 class
# -------------------------
class M2N2:
    """
    M2N2 evolutionary algorithm.

    Key hyperparameters:
     - alpha: competition strength (paper Eq.5). alpha=1 equals standard implicit sharing.
     - use_mutation: whether to apply Gaussian parameter noise after crossover (used when evolving from scratch).
     - mutation_std: standard deviation for gaussian mutation.
     - cache_on_device: whether to copy the entire training set to device for fast per-eval forward pass.
    """
    def __init__(self, model_class, train_loader, test_loader, archive_size=20,
                 device='cuda', alpha=1.0, use_mutation=True, mutation_std=0.01,
                 cache_on_device=True, seed=42, capacity_mode='binary'):
        """
        capacity_mode:
         - 'binary': c_j = 1 for MNIST with binary scoring (correct/incorrect).
         - 'max_score': c_j = max_i s(x_j | theta_i) used for continuous rewards (paper description).
        """
        self.model_class = model_class
        self.test_loader = test_loader
        self.archive_size = archive_size
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.epsilon = 1e-8
        self.alpha = float(alpha)
        self.use_mutation = use_mutation
        self.mutation_std = float(mutation_std)
        self.cache_on_device = bool(cache_on_device)
        self.seed = int(seed)
        self.capacity_mode = capacity_mode

        # seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --- load and optionally cache train data ---
        print("Loading training data into memory (will be cached on device if configured)...")
        imgs = []
        lbls = []
        for batch in train_loader:
            imgs.append(batch[0])
            lbls.append(batch[1])
        self.train_images = torch.cat(imgs, dim=0)
        self.train_labels = torch.cat(lbls, dim=0)
        self.num_train_samples = len(self.train_labels)

        if self.cache_on_device:
            self.train_images = self.train_images.to(self.device)
            self.train_labels = self.train_labels.to(self.device)

        self.forward_passes = 0

        # --- initialize archive ---
        print(f"Initializing archive with {archive_size} random models...")
        self.archive = [self.model_class().to(self.device) for _ in range(archive_size)]
        # scores: per-model, per-sample correctness/score (archive_size, N)
        self.scores = torch.zeros(self.archive_size, self.num_train_samples, device=self.device, dtype=torch.float32)
        self.fitness = torch.zeros(self.archive_size, device=self.device, dtype=torch.float32)
        self.best_model = None

    @torch.no_grad()
    def _evaluate_model(self, model):
        """Evaluate per-sample scores (1 for correct, 0 for wrong) or continuous scores if provided.
           Returns tensor shape (N,) on self.device.
        """
        model = model.to(self.device)
        model.eval()
        if self.cache_on_device:
            outputs = model(self.train_images)
            preds = outputs.argmax(dim=1)
            scores = (preds == self.train_labels).float()
            self.forward_passes += 1
            return scores
        else:
            # batch-wise eval to avoid caching
            batch_size = 256
            pieces = []
            for i in range(0, self.num_train_samples, batch_size):
                imgs = self.train_images[i:i+batch_size].to(self.device)
                lbls = self.train_labels[i:i+batch_size].to(self.device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                pieces.append((preds == lbls).float().cpu())
            self.forward_passes += 1
            return torch.cat(pieces, dim=0).to(self.device)

    def _compute_capacity(self):
        """Compute c_j according to the selected capacity_mode."""
        if self.capacity_mode == 'binary':
            return torch.ones(self.num_train_samples, device=self.device, dtype=torch.float32)
        elif self.capacity_mode == 'max_score':
            # c_j = max_i s(x_j | theta_i)
            return self.scores.max(dim=0).values.clone().to(self.device)
        else:
            raise ValueError("Unknown capacity_mode: choose 'binary' or 'max_score'")

    def _update_fitness(self):
        """
        Update fitness using the paper's modified optimization goal (Eq. 3 and Eq. 5):
            f(theta_i) = sum_j [ s(x_j | theta_i) * c_j / (z_j ** alpha + eps) ]
        where z_j = sum_k s(x_j | theta_k)
        """
        z_j = self.scores.sum(dim=0) + self.epsilon  # (N,)
        c_j = self._compute_capacity()               # (N,)
        # elementwise division; result per-model then summed over samples
        denom = (z_j ** self.alpha) + self.epsilon
        # safe broadcast: (archive_size, N) * (N,) -> (archive_size, N)
        self.fitness = ((self.scores * c_j) / denom).sum(dim=1)
        self.fitness = torch.nan_to_num(self.fitness, nan=0.0, posinf=0.0, neginf=0.0)

        # sort archive by descending fitness
        sorted_idx = torch.argsort(self.fitness, descending=True)
        self.archive = [self.archive[i] for i in sorted_idx]
        self.scores = self.scores[sorted_idx]
        self.fitness = self.fitness[sorted_idx]

    def _select_parents(self):
        """
        Parent A: sampled proportional to fitness (with safe fallback to uniform).
        Parent B: sampled based on attraction g(θA, θB) (paper Eq.4):
           g(θA, θB) = sum_j [ c_j / (z_j + eps) * max(s_Bj - s_Aj, 0) ]
        """
        # Parent A selection
        fitness_sum = float(self.fitness.sum().cpu().item())
        if fitness_sum <= 0 or torch.isnan(self.fitness).any():
            probs_a = np.ones(self.archive_size) / self.archive_size
        else:
            probs_a = (self.fitness / self.fitness.sum()).cpu().numpy()
            probs_a = np.clip(probs_a, 1e-12, None)
            probs_a = probs_a / probs_a.sum()
        parent_a_idx = int(np.random.choice(self.archive_size, p=probs_a))
        parent_a = self.archive[parent_a_idx]

        # Attraction for Parent B
        z_j = self.scores.sum(dim=0) + self.epsilon
        c_j = self._compute_capacity()  # (N,)
        parent_a_scores = self.scores[parent_a_idx]  # (N,)

        complementarity = F.relu(self.scores - parent_a_scores)  # (archive_size, N)
        # multiply per-sample by c_j / (z_j + eps)
        factor = (c_j / (z_j + self.epsilon)).unsqueeze(0)  # (1, N)
        attraction_scores = (complementarity * factor).sum(dim=1)  # (archive_size,)

        # prevent self-selection
        attraction_scores = attraction_scores.clone()
        attraction_scores[parent_a_idx] = -float('inf')

        if torch.isfinite(attraction_scores).sum() == 0 or (attraction_scores <= 0).all():
            # fallback: choose a random different partner
            choices = [i for i in range(self.archive_size) if i != parent_a_idx]
            parent_b_idx = int(np.random.choice(choices))
        else:
            probs_b = F.softmax(attraction_scores, dim=0).cpu().numpy()
            probs_b = np.clip(probs_b, 1e-12, None)
            probs_b = probs_b / probs_b.sum()
            parent_b_idx = int(np.random.choice(self.archive_size, p=probs_b))

        parent_b = self.archive[parent_b_idx]
        return parent_a, parent_b

    def _crossover(self, parent_a, parent_b):
        """
        Merge parents using random split-point and SLERP on each side (paper Sec. 3.1).
        After setting params, optionally apply Gaussian mutation (used when evolving from scratch).
        """
        p1 = get_params(parent_a, device=self.device, dtype=next(parent_a.parameters()).dtype)
        p2 = get_params(parent_b, device=self.device, dtype=next(parent_b.parameters()).dtype)
        num_params = len(p1)
        if num_params == 0:
            return self.model_class().to(self.device)

        split_point = np.random.randint(1, num_params) if num_params > 1 else 1
        mix_ratio = float(np.random.rand())

        # split and SLERP
        part1 = slerp(p1[:split_point], p2[:split_point], mix_ratio)
        part2 = slerp(p1[split_point:], p2[split_point:], 1.0 - mix_ratio)
        child_vector = torch.cat([part1, part2]).to(self.device)

        child = self.model_class().to(self.device)
        set_params(child, child_vector)

        # mutation: gaussian noise added to parameters
        if self.use_mutation and self.mutation_std > 0.0:
            with torch.no_grad():
                for p in child.parameters():
                    p.add_(torch.randn_like(p) * self.mutation_std)

        return child

    @torch.no_grad()
    def test_model(self, model):
        """Standard network evaluation on the test_loader. Returns accuracy percentage."""
        model = model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total if total > 0 else 0.0

    def evolve(self, num_evaluations=50000, log_interval=1000):
        """
        Main evolutionary loop. Each evaluation == one forward pass of the full cached train set.
        This matches the paper's evaluation counting (they report # forward passes).
        """
        # initial evaluation of archive
        for i in range(self.archive_size):
            self.scores[i] = self._evaluate_model(self.archive[i])

        pbar = tqdm(total=num_evaluations, initial=self.forward_passes, desc="Evolving Models")
        while self.forward_passes < num_evaluations:
            # update fitness + sort archive
            self._update_fitness()

            # select parents & produce child
            parent_a, parent_b = self._select_parents()
            child = self._crossover(parent_a, parent_b)

            # evaluate child on training set
            child_scores = self._evaluate_model(child)

            # compute child's fitness under current ecosystem (z_j from current scores)
            current_z_j = self.scores.sum(dim=0) + self.epsilon
            c_j = self._compute_capacity()
            denom = (current_z_j ** self.alpha) + self.epsilon
            child_fitness = ((child_scores * c_j) / denom).sum()
            child_fitness = float(torch.nan_to_num(child_fitness, nan=0.0, posinf=0.0, neginf=0.0).cpu().item())

            worst_fitness = float(self.fitness[-1].cpu().item())

            # if child better than worst -> replace
            if child_fitness > worst_fitness:
                self.archive[-1] = child
                self.scores[-1] = child_scores
                # fitness reorder will happen next iteration via _update_fitness()

            # periodic logging & snapshot of best model
            if self.forward_passes % log_interval == 0:
                self._update_fitness()
                self.best_model = deepcopy(self.archive[0])
                test_acc = self.test_model(self.best_model)
                pbar.set_postfix({"best_fitness": f"{self.fitness[0]:.4f}", "test_acc": f"{test_acc:.2f}%"})

            pbar.update(1)

        pbar.close()
        print("\nEvolution finished.")
        self._update_fitness()
        self.best_model = deepcopy(self.archive[0])
        final_acc = self.test_model(self.best_model)
        print(f"Final Test Accuracy of the best model: {final_acc:.2f}%")
        return final_acc

# -------------------------
# Main execution
# -------------------------
if __name__ == '__main__':
    # --- Configuration: these follow the paper defaults for the "from scratch" MNIST experiment ---
    ARCHIVE_SIZE = 20
    BATCH_SIZE = 1024         # for faster eval, matches your original script
    NUM_EVALUATIONS = 50000   # number of forward passes (paper reports evaluations in forward passes)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

    print(f"Using device: {DEVICE}")

    pl.seed_everything(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Data
    mnist_dm = MNISTDataModule(batch_size=BATCH_SIZE)
    mnist_dm.prepare_data()
    mnist_dm.setup('fit')
    mnist_dm.setup('test')

    # Create evolver
    evolver = M2N2(
        model_class=MLP,
        train_loader=mnist_dm.train_dataloader(),
        test_loader=mnist_dm.test_dataloader(),
        archive_size=ARCHIVE_SIZE,
        device=DEVICE,
        alpha=1.0,             # try 0.0, 0.5, 1.0, >1 to reproduce Fig.4 trends
        use_mutation=True,     # when evolving from scratch paper uses mutation (Gaussian noise)
        mutation_std=0.01,     # suggested small std; tune if needed
        cache_on_device=True,  # set False if you run out of GPU memory
        seed=SEED,
        capacity_mode='binary' # 'binary' for MNIST (c_j = 1). Use 'max_score' for continuous rewards.
    )

    # Run evolution
    evolver.evolve(num_evaluations=NUM_EVALUATIONS, log_interval=1000)
