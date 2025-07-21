from abc import ABC, abstractmethod

import numpy as np

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
    def __init__(self) -> None:
        super().__init__()
        self._weights = None

    def partial_fit(self, X, y, groups: np.ndarray=None):
        if groups is None:
            groups = np.zeros(len(X))
            
        unique_groups = np.unique(groups)
        for group_id in unique_groups:
            mask = groups == group_id
            if np.sum(mask) == 0:
                continue

            if len(X[mask].shape) == 3 and len(y[mask].shape) == 2:
                X_group = X[mask].reshape((X[mask].shape[0] * X[mask].shape[1], X[mask].shape[2]))
                y_group = y[mask].reshape((y[mask].shape[0] * y[mask].shape[1]))
            elif len(X[mask].shape) > 3:
                raise RuntimeError("Cannot handle data with more than 3dim.")
            else:
                X_group, y_group = X[mask], y[mask]
            self._partial_fit_impl(X_group, y_group)
            
        return self

    def evaluate(self):
        if self._weights is None:
            raise RuntimeError("Model has not been trained yet. Call `partial_fit` to train the model.")
        self._evaluate_impl()

    @abstractmethod
    def _partial_fit_impl(self, X, y):
        pass

    @abstractmethod
    def predict(self, X, groups=None):
        pass

    @abstractmethod
    def _evaluate_impl(self):
        """Evaluation of training data after calling fit"""
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
