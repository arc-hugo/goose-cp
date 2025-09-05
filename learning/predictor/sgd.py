import logging
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics.functional import r2_score

from .base_predictor import BaseCPPredictor

class LinearSoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(LinearSoftmaxModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1, dtype=torch.float64, bias=False)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Appliquer les couches linéaires position par position
        x = self.linear1(x)
        
        # Supprimer la dernière dimension
        x = x.squeeze(-1)  # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        x = F.softmax(x, dim=1)
        
        return x


class ConvSoftmaxModel(nn.Module):
    def __init__(self, input_dim, kernel_size=128):
        super(ConvSoftmaxModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 1, kernel_size=3, padding=1, dtype=torch.float64, bias=False)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Appliquer les couches de convolution position par position
        x = self.conv1(x)
        
        # Supprimer la dernière dimension
        x = x.squeeze(-1)  # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        # x = F.softmax(x, dim=1)
        
        return x

class SGDRegressorSoftmax(BaseCPPredictor):
    def __init__(self, input_dim: int, criterion=nn.CrossEntropyLoss(),
                 optimizer=torch.optim.Adam, epoch=3, alpha=1e-3):
        super().__init__(epoch=epoch)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = LinearSoftmaxModel(input_dim)
        self._model.to(self._device)

        self._fitted = False

        self.criterion = criterion
        self.optimizer = optimizer(self._model.parameters(), alpha)

    def _train_impl(self, data: DataLoader):
        self._model.train()

        for batch, (X, y) in enumerate(data):
            # Pass data to device
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction and loss
            pred = self._model(X)
            loss = self.criterion(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f}")
        
        self._fitted = True
    
    def _evaluate_impl(self, data: DataLoader):
        self._model.eval()

        test_loss, num_batches, score = 0, 0, 0
    
        with torch.no_grad():
            for X, y in data:
                # Pass data to device
                X, y = X.to(self._device), y.to(self._device)
                
                pred = self._model(X)

                test_loss += self.criterion(pred, y).item()
                score += r2_score(pred.squeeze(), y.squeeze())
                num_batches += 1

        score /= num_batches
        test_loss /= num_batches
        logging.info(f"Test Error: Average R2 score: {score}, Avg loss: {test_loss:>8f}")

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
        

