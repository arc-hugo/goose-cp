import os
import glob
import json
from abc import abstractmethod
from enum import Enum

import toml
import numpy as np
from tqdm import tqdm

import wlplan
from learning.dataset.container.base_dataset import Dataset
from util.error_message import get_path_error_msg
from wlplan.feature_generation import Features
from wlplan.planning import Variable

MAX_EXPANSIONS_PER_PROBLEM = 10000
MAX_STATE_SPACE_DATA = 100000


class ProblemType(Enum):
    Plans = 1
    StateSpace = 2
    CostPartition = 3


class CostPartitionData:
    def __init__(self,
                 folder_path : str,
                 problem_path: str,
                 patterns_path: str,
                 data_paths: list[str]):
        self._folder_path = folder_path
        self.problem = problem_path
        self.patterns = patterns_path
        self.datas = data_paths
    
    def parse_patterns(self) -> list[list[int]]:
        with open(self.patterns) as f:
            json_f = json.load(f)

            return json_f["pattern"]
    
    def parse_data(self) -> tuple[list[list[int]], list[np.ndarray[float]]]:
        X = []
        y = []

        for data in self.datas:
            with open(data) as f:
                json_f = json.load(f)

                for entry in json_f:
                    y.append(entry["costs"])

                    assign = []
                    for i, val in enumerate(entry["state"]):
                        assign.append(Variable("var"+str(i), i, val))

                    X.append(assign)
        
        return X, y


class DatasetCreator:
    def __init__(
        self,
        data_config: str,
        feature_generator: Features,
        hash_prefix: str,
    ):
        # domain information
        data_config = toml.load(data_config)

        self.domain_pddl = data_config["domain_pddl"]
        self.tasks_dir = data_config["tasks_dir"]
        # plans_dir collected later as not always necessary (e.g. state space data)
        self._data_config = data_config

        assert os.path.exists(self.domain_pddl), get_path_error_msg(self.domain_pddl)
        assert os.path.exists(self.tasks_dir), get_path_error_msg(self.tasks_dir)

        self.wlplan_domain = wlplan.planning.parse_domain(self.domain_pddl)

        # feature generator
        self.feature_generator = feature_generator

        # prevent tmp files from being overwritten by parallel jobs
        self.hash_prefix = hash_prefix

    def _get_problem_iterator(self, problem_type: ProblemType = ProblemType.Plans):
        pbar = []
        match problem_type:
            case ProblemType.Plans:
                self.plans_dir = self._data_config["plans_dir"]
                assert os.path.exists(self.plans_dir), get_path_error_msg(self.plans_dir)
                for f in sorted(os.listdir(self.plans_dir)):
                    problem_pddl = self.tasks_dir + "/" + f.replace(".plan", ".pddl")
                    plan_file = self.plans_dir + "/" + f
                    pbar.append((problem_pddl, plan_file))
            case ProblemType.StateSpace:
                pbar = [self.tasks_dir + "/" + f for f in sorted(os.listdir(self.tasks_dir))]
            case ProblemType.CostPartition:
                problem_dirs = [f for f in os.scandir(self.tasks_dir) if f.is_dir()]
                for dir in problem_dirs:
                    data_files = sorted(glob.glob(dir.path + "/data-*.json"))

                    needed_files = (os.path.exists(dir.path + "/problem.pddl") and 
                                    os.path.exists(dir.path + "/pattern.json") and
                                    len(data_files) > 0)
                    
                    if not needed_files:
                        continue

                    pbar.append(CostPartitionData(dir.path, dir.path + "/problem.pddl",
                                                  dir.path + "/pattern.json",
                                                  data_files))

        pbar = tqdm(pbar, desc="Collecting data from problems")
        return pbar

    @abstractmethod
    def get_dataset(self) -> Dataset:
        pass
