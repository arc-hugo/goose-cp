from abc import ABC, abstractmethod

import comet_ml
from comet_ml.integration.pytorch import log_model

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
    def __init__(self, domain: str, action_schema: str, iterations: int, epoch = 10, alpha=1e-3, opt_params={}) -> None:
        super().__init__()
        self._weights = None
        self.epochs = epoch

        self.params = {
            "learning_rate": alpha,
            "domain": domain,
            "action_schema": action_schema,
            "iterations": iterations,
            "epochs": self.epochs,
            **opt_params
        }

    def fit(self, train_data: DataLoader, validation_data: DataLoader):
        exp = comet_ml.start(
            project_name="goose-cp"
        )

        exp.disable_mp()

        exp.log_parameters(self.params)
        
        
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            with exp.train():
                self._train_impl(train_data, t, exp)

            with exp.test():
                self._validate_impl(validation_data, t, exp)
        
        log_model(exp, self.get_model(), "LinearSoftmaxModel-CP")

        exp.end()
        
        return self

    @abstractmethod
    def _train_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _validate_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment):
        """Evaluation of training data after calling fit"""
        pass

    @abstractmethod
    def get_model(self):
        """Get model object"""
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
