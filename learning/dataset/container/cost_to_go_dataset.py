from learning.dataset.container.base_dataset import Dataset
from wlplan.data import ProblemStates, LiftedDataset
from wlplan.planning import Domain


class CostToGoDataset(Dataset):
    def __init__(self, wlplan_domain: Domain, data: list[ProblemStates], y: list[float]):
        dataset = LiftedDataset(wlplan_domain, data)
        self._y = y
        self._data = data
        
        super().__init__(wlplan_domain, dataset)

    @property
    def y(self) -> list[float]:
        return self._y

    @property
    def data(self) -> list[ProblemStates]:
        return self._data
    
    def __len__(self):
        return len(self._data)