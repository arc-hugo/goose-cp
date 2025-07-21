import numpy as np

from learning.dataset.container.base_dataset import Dataset
from wlplan.data import ProblemPatternsAssignments, GroundedDataset
from wlplan.planning import Domain

class CostPartitionDataset(Dataset):
    def __init__(self, wlplan_domain: Domain, data: list[ProblemPatternsAssignments], y: list[np.matrix[float]]):
        dataset = GroundedDataset(wlplan_domain, data)
        self._y = y
        self._data = data
        
        super().__init__(wlplan_domain, dataset)

    @property
    def y(self) -> list[dict[str,float]]:
        return self._y

    @property
    def data(self) -> list[ProblemPatternsAssignments]:
        return self._data
    
    def __len__(self):
        return len(self._data)