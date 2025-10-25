import logging
import numpy as np
import torch
import comet_ml

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base_predictor import BaseCPPredictor

class BRNNSoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_hidden=1):
        super(BRNNSoftmaxModel, self).__init__()

        self._hidden_dim = hidden_dim
        self._num_hidden = num_hidden

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_hidden, 
                          nonlinearity="relu", batch_first=True, bidirectional=True, dtype=torch.float64)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_dim, 1, dtype=torch.float64)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)

        # hidden states
        h0 = torch.zeros(self._num_hidden * 2, x.size(0), self._hidden_dim, dtype=torch.float64).to(x.device)
        
        # Appliquer les couches linéaires
        x, _ = self.rnn(x, h0)
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Supprimer la dernière dimension
        x = x.squeeze(-1)  # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        x = F.log_softmax(x, dim=1)
        
        return x

class BRNNSoftmax(BaseCPPredictor):
    def __init__(self, input_dim: int, domain: str, action_schema: str, iterations: int,
                 criterion=nn.KLDivLoss, optimizer=Adam, scheduler=CosineAnnealingLR,
                 epoch=1000, alpha=1e-5, device="cuda:0"):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = BRNNSoftmaxModel(input_dim)
        self._model.to(self._device)

        self._fitted = False

        self.criterion = criterion(reduction='batchmean')
        self.optimizer = optimizer(self._model.parameters(), alpha)
        self.scheduler = scheduler(self.optimizer, 100)

        opt_params = {
            "criterion": self.criterion.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "hidden_dim": self._model._hidden_dim,
            "num_hidden": self._model._num_hidden
        }

        super().__init__(domain, action_schema, iterations, epoch=epoch, alpha=alpha, opt_params=opt_params)

    def _train_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment) -> float:
        self._model.train()
        total_loss = 0
        nb_data = 0
        # first = True

        for train_data in data:
            for X,y in train_data:
                # Pass data to device
                X, y = X.to(self._device), y.to(self._device)

                # Compute prediction and loss
                pred = self._model(X)

                # if first:
                #     print(X)
                #     print(pred)
                #     print(y)
                #     first = False
                
                loss = self.criterion(pred, y)

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                total_loss += loss.item()
                nb_data += 1
        
        total_loss /= nb_data
        exp.log_metrics({"loss": total_loss}, epoch=epoch)

        logging.info(f"Avg loss: {total_loss:>8f}")
        self._fitted = True

        return total_loss
    
    def _validate_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment) -> float:
        self._model.eval()

        total_loss, num_batches = 0, 0
    
        with torch.no_grad():
            for train_data in data:
                for X,y in train_data:
                    # Pass data to device
                    X, y = X.to(self._device), y.to(self._device)
                    
                    pred = self._model(X)

                    total_loss += self.criterion(pred, y).item()
                    num_batches += 1

        total_loss /= num_batches

        exp.log_metrics({"loss": total_loss}, epoch=epoch)
        logging.info(f"Avg validation loss: {total_loss:>8f}")

        return total_loss

    def get_model(self):
        return self._model

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
        

