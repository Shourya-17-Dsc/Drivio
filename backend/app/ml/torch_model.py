"""Small PyTorch MLP with a scikit-learn style fit/predict interface."""
from __future__ import annotations

import numpy as np


class TorchMLPClassifier:
    def __init__(self, hidden: int = 32, epochs: int = 200, lr: float = 1e-2, seed: int = 0) -> None:
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, len(self.classes_)))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y_idx, dtype=torch.long)
        for _ in range(self.epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(self.net(xt), yt).backward()
            opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32))
        return self.classes_[logits.argmax(dim=1).numpy()]
