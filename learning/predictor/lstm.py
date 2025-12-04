import logging
import torch
import comet_ml

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base_predictor import BaseEpochPredictor

class LSTMSoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_hidden=2):
        super(LSTMSoftmaxModel, self).__init__()

        self._hidden_dim = hidden_dim
        self._num_hidden = num_hidden

        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_hidden, dropout=0.25,
                          batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)

        # hidden states and cell states
        h0 = torch.zeros(self._num_hidden * 2, x.size(0), self._hidden_dim).to(x.device)
        c0 = torch.zeros(self._num_hidden * 2, x.size(0), self._hidden_dim).to(x.device)
        
        # Appliquer les couches linéaires
        x, _ = self.rnn(x, (h0, c0))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Supprimer la dernière dimension
        x = x.squeeze(-1)  # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        x = F.log_softmax(x, dim=1)
        
        return x

class LSTMSoftmax(BaseEpochPredictor):
    def __init__(self, input_dim: int, domain: str, action_schema: str, iterations: int,
                 criterion=nn.KLDivLoss, optimizer=Adam,
                 epoch=3000, alpha=1e-5, device="cuda:0"):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = LSTMSoftmaxModel(input_dim)
        self._model.to(self._device)

        self._fitted = False

        self.criterion = criterion(reduction='batchmean')
        self.optimizer = optimizer(self._model.parameters(), alpha)

        opt_params = {
            "model": "LSTMSoftmax",
            "criterion": self.criterion.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "hidden_dim": self._model._hidden_dim,
            "num_hidden": self._model._num_hidden
        }

        super().__init__(domain, action_schema, iterations, epoch=epoch, alpha=alpha, log_params=opt_params)

    def _train_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment) -> float:
        self._model.train()
        total_loss = 0
        nb_data = 0
        first = True

        for train_data in data:
            for X,y in train_data:
                # Pass data to device
                X, y = X.to(self._device), y.to(self._device)

                # Compute prediction and loss
                pred = self._model(X)

                if first:
                    print(pred)
                    print(y)
                    first = False
                
                loss = self.criterion(pred, y)

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

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

    def get_model(self) -> nn.Module:
        return self._model
    

        

