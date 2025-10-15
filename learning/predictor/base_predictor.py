from abc import ABC, abstractmethod

import wandb

import numpy as np

from torch.utils.data import DataLoader

from util.custom_decorators import abstract_class_attributes
from util.timer import TimerContextManager


@abstract_class_attributes("IS_RANK")
class BasePredictor(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._weights = None

    def fit_evaluate(self, X, y, sample_weight):
        with TimerContextManager("training model"):
            self.fit(X, y, sample_weight)
        with TimerContextManager("evaluating model"):
            self.evaluate()

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self._fit_impl(X, y, sample_weight)

    def evaluate(self):
        if self._weights is None:
            raise RuntimeError("Model has not been trained yet. Call `fit` to train the model.")
        self._evaluate_impl()

    @abstractmethod
    def _fit_impl(self, X, y, sample_weight):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _evaluate_impl(self):
        """Evaluation of training data after calling fit"""
        pass

    def get_weights(self) -> list:
        if self._weights is None:
            raise RuntimeError("Model has not been trained yet. Call `fit` to train the model.")
        ret = self._weights
        if isinstance(ret, np.ndarray):
            ret = ret.tolist()
        return ret

class BaseCPPredictor(ABC):
    def __init__(self, domain: str, action_schema: str, epoch = 1, alpha=1e-3, opt_config={}) -> None:
        super().__init__()
        self._weights = None
        self.epochs = epoch
        self.alpha = alpha
        self.domain = domain
        self.action_schema = action_schema
        self.opt_config = opt_config

    def fit(self, data: DataLoader):
        run = wandb.init(
            entity="corail",
            project="goose-cp",
            config={
                "learning_rate": self.alpha,
                "domain": self.domain,
                "action_schema": self.action_schema,
                "epochs": self.epochs
            } + self.opt_config
        )

        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self._train_impl(data)
            self._evaluate_impl(data, run)

        # self._save_weights()
        
        run.finish()
        
        return self

    @abstractmethod
    def _train_impl(self, data: DataLoader):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _evaluate_impl(self, data: DataLoader, run: wandb.Run):
        """Evaluation of training data after calling fit"""
        pass

    @abstractmethod
    def _save_weights(self):
        """Save weights after fit"""
        pass

    def get_weights(self) -> list:
        if self._weights is None:
            raise RuntimeError("Model has not been trained yet. Call `partial_fit` to train the model.")
        ret = self._weights
        if isinstance(ret, np.ndarray):
            ret = ret.tolist()
        return ret

    @abstractmethod
    def set_weights(self, weights):
        pass
