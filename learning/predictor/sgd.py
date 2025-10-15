import logging
import numpy as np
import torch
import wandb

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base_predictor import BaseCPPredictor

class LinearSoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(LinearSoftmaxModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, hidden_dim, dtype=torch.float64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, 1, dtype=torch.float64)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Appliquer les couches linéaires position par position
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        
        # Supprimer la dernière dimension
        x = x.squeeze(-1)  # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        x = F.softmax(x, dim=1)
        
        return x

class RegressorSoftmax(BaseCPPredictor):
    def __init__(self, input_dim: int, domain: str, action_schema: str,
                 criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, epoch=10, alpha=1e-3,
                 device="cuda:0"):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = LinearSoftmaxModel(input_dim)
        self._model.to(self._device)

        self._fitted = False

        self.criterion = criterion()
        self.optimizer = optimizer(self._model.parameters(), alpha)

        opt_config = {
            "criterion": self.criterion.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__
        }

        super().__init__(domain, action_schema, epoch=epoch, alpha=alpha, opt_config=opt_config)

    def _train_impl(self, data: DataLoader):
        self._model.train()

        for _, (X, y) in enumerate(data):
            # Pass data to device
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction and loss
            pred = self._model(X)
            loss = self.criterion(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # loss = loss.item()
            # print(f"loss: {loss:>7f}")
        
        self._fitted = True
    
    def _evaluate_impl(self, data: DataLoader, run: wandb.Run):
        self._model.eval()

        test_loss, num_batches = 0, 0
    
        with torch.no_grad():
            for X, y in data:
                # Pass data to device
                X, y = X.to(self._device), y.to(self._device)
                
                pred = self._model(X)

                test_loss += self.criterion(pred, y).item()
                num_batches += 1

        test_loss /= num_batches
        logging.info(f"Avg loss: {test_loss:>8f}")
        run.log({"loss": test_loss})


    def _save_weights(self):
        self._weights = self._model.state_dict()["linear1.weight"].squeeze().tolist()

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model has not been trained yet. Call `fit` to train the model.")

        self._model.eval()
        return self._model.forward(X)
    
    def set_weights(self, weights):
        self._weights = np.array(weights)
        self._model.coef_ = np.array(weights)
        self._model.intercept_ = np.zeros((1,))
        self._fitted = True
        

