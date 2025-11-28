import random
import torch

import numpy as np

from torch import Tensor
from torch.utils.data import IterableDataset

from learning.dataset.container.cost_partition_dataset import CostPartitionDataset
from learning.dataset.container.base_dataset import Dataset

from wlplan.feature_generation import CostPartitionFeatures, Features

class CustomIterableDataset(IterableDataset):
    def __init__(self, feature_generator: CostPartitionFeatures, dataset: Dataset):
        super().__init__()
        self.fg: CostPartitionFeatures = feature_generator
        self.data: Dataset = dataset
        self.use_cache = False
        self.cache = {}

        self.embed_dataset_fn = self.fg.graph_and_actions_embed_dataset


    def purge_cache(self):
        del self.cache
        self.cache = {}
        self.use_cache = False

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        if not self.use_cache:
            for i, input in enumerate(self.embed_dataset_fn(self.data.wlplan_dataset)):
                for action_name in input:
                    if (action_name in self.data.y[i]):
                        X = torch.from_numpy(np.array(input[action_name]))
                        y = torch.from_numpy(self.data.y[i][action_name])

                        if X.shape not in self.cache.keys():
                            self.cache[X.shape] = []
                        self.cache[X.shape].append((X,y))

                        yield X,y

            self.use_cache = True
        else:
            sequences = list(self.cache.keys())
            random.shuffle(sequences)
            for seq in sequences:
                random.shuffle(self.cache[seq])
                for train_data in self.cache[seq]:
                    yield train_data

class ActionSchemaIterableDataset(IterableDataset):
    def __init__(self, feature_generator: CostPartitionFeatures, dataset: Dataset, action_schema: str):
        super().__init__()
        self.fg: CostPartitionFeatures = feature_generator
        self.data: Dataset = dataset
        self.use_cache = False
        self.action_schema = action_schema
        self.cache = {}

        self.embed_dataset_fn = self.fg.graph_and_actions_embed_dataset


    def purge_cache(self):
        del self.cache
        self.cache = {}
        self.use_cache = False

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        if not self.use_cache:
            for i, input in enumerate(self.embed_dataset_fn(self.data.wlplan_dataset)):
                for action_name in input:
                    action_schema = get_action_schema_name(action_name)
                    if (action_schema == self.action_schema and action_name in self.data.y[i]):
                        X = torch.from_numpy(np.array(input[action_name]))
                        y = torch.from_numpy(self.data.y[i][action_name])

                        if X.shape not in self.cache.keys():
                            self.cache[X.shape] = []
                        self.cache[X.shape].append((X,y))

                        yield X,y

            self.use_cache = True
        else:
            sequences = list(self.cache.keys())
            random.shuffle(sequences)
            for seq in sequences:
                random.shuffle(self.cache[seq])
                for train_data in self.cache[seq]:
                    yield train_data

def get_action_schema_name(name: str):
    return name.split(" ")[0]

def get_action_schemas_data(dataset: Dataset, feature_generator: Features) -> list[ActionSchemaIterableDataset]:
    assert isinstance(dataset, CostPartitionDataset)
    assert isinstance(feature_generator, CostPartitionFeatures)

    return [ActionSchemaIterableDataset(feature_generator, dataset, action_schema.name) for action_schema in dataset.domain.action_schemas]

def get_all_schemas_data(dataset: Dataset, feature_generator: Features) -> CustomIterableDataset:
    assert isinstance(dataset, CostPartitionDataset)
    assert isinstance(feature_generator, CostPartitionFeatures)

    return CustomIterableDataset(feature_generator, dataset)

def collate_variable_seq(batch: list[tuple[Tensor, Tensor]]):
    new_batch = []
    seq_size = None
    stack_begin = 0

    X,Y = zip(*batch)
    for i, (x, _) in enumerate(zip(X,Y)):
        if seq_size is None:
            seq_size = x.shape
        elif seq_size != x.shape:
            seq_size = x.shape
            stacked_X = torch.stack(X[stack_begin:i])
            stacked_y = torch.stack(Y[stack_begin:i])

            new_batch.append((stacked_X, stacked_y))
            stack_begin = i

    if stack_begin < len(batch):
        stacked_X = torch.stack(X[stack_begin:len(batch)])
        stacked_y = torch.stack(Y[stack_begin:len(batch)])

        new_batch.append((stacked_X, stacked_y))
    
    return new_batch

