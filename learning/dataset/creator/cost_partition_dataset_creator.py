import os
import glob

from abc import abstractmethod

from learning.dataset.container.cost_partition_dataset import CostPartitionDataset
from learning.dataset.creator.dataset_creator import DatasetCreator, CostPartitionData, ProblemType

from wlplan.feature_generation import CostPartitionFeatures
from wlplan.planning import parse_grounded_problem, parse_domain
from wlplan.data import ProblemPatternsAssignments

class CostPartitionDatasetCreator(DatasetCreator):
    """Base class for creating datasets for cost partition learning."""

    def __init__(
        self,
        data_config: str,
        feature_generator: CostPartitionFeatures,
        hash_prefix: str,
    ):
        super().__init__(
            data_config=data_config,
            feature_generator=feature_generator,
            hash_prefix=hash_prefix,
        )

        if not isinstance(feature_generator, (CostPartitionFeatures)):
            raise ValueError("Cost partition datasets must use dedicated features generators")

        if "validation_dir" not in self._data_config:
            raise ValueError("Must specify a directory with validation files")
        
        self.domain = parse_domain(self.domain_pddl, keep_statics=False)
        self.validation_dir = self._data_config["validation_dir"]

    @abstractmethod
    def get_dataset(self) -> CostPartitionDataset:
        wlplan_data = []
        y = []

        data: CostPartitionData
        for data in self._get_problem_iterator(ProblemType.CostPartition):
            # Parse grounded problem
            grounded_problem = parse_grounded_problem(self.domain_pddl, data.problem)

            # Parse patterns
            patterns = data.parse_patterns()

            # Parse assignments and costs
            X_i, y_i = data.parse_data()

            ppa = ProblemPatternsAssignments(grounded_problem, patterns, X_i)

            wlplan_data.append(ppa)
            y.extend(y_i)

        dataset = CostPartitionDataset(self.domain, wlplan_data, y)
        return dataset

    def get_validation_dataset(self):
        wlplan_data = []
        y = []

        problem_dirs = [f for f in os.scandir(self.validation_dir) if f.is_dir()]
        for dir in problem_dirs:
            data_files = sorted(glob.glob(dir.path + "/data-*.json"))
            problem_file = dir.path + "/problem.pddl"
            pattern_file = dir.path + "/pattern.json"

            needed_files = (os.path.exists(problem_file) and os.path.exists(pattern_file)
                            and len(data_files) > 0)
            
            if not needed_files:
                continue
            
            data = CostPartitionData(dir.path, problem_file, pattern_file, data_files)

            grounded_problem = parse_grounded_problem(self.domain_pddl, problem_file)

            patterns = data.parse_patterns()

            X_i, y_i = data.parse_data()

            ppa = ProblemPatternsAssignments(grounded_problem, patterns, X_i)

            wlplan_data.append(ppa)
            y.extend(y_i)

        dataset = CostPartitionDataset(self.domain, wlplan_data, y)
        return dataset


