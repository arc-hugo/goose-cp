from abc import ABC, abstractmethod

from wlplan.data import Dataset as WLPlanDataset
from wlplan.planning import Domain


class Dataset(ABC):
    def __init__(self, wlplan_domain: Domain, wlplan_dataset: WLPlanDataset):
        self._domain = wlplan_domain
        self._wlplan_dataset = wlplan_dataset

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def wlplan_dataset(self) -> WLPlanDataset:
        return self._wlplan_dataset

    @property
    @abstractmethod
    def y(self):
        pass

    @abstractmethod
    def __len__(self):
        pass