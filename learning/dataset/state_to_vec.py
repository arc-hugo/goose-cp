from argparse import Namespace

import random
# import math

import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import IterableDataset, default_collate

from learning.dataset.container.base_dataset import Dataset
from learning.dataset.container.ranking_dataset import RankingDataset
from learning.dataset.container.cost_partition_dataset import CostPartitionDataset
from wlplan.feature_generation import Features, CostPartitionFeatures

class SameSizeCollate:
    def __init__(self) -> None:
        self.save_batch = None

    def collate_fn(self, batch):
        if self.save_batch is None:
            # print(batch[0][0].size)
            size = batch[0][1].size
            for i in range(1, len(batch)):
                if batch[i][1].size != size:
                    self.save_batch = batch[i:]
                    return default_collate(batch[:i])
            
            return default_collate(batch)

        size = self.save_batch[0][1].size
        for i in range(1, len(self.save_batch)):
            if self.save_batch[i][1].size != size:
                out_batch = self.save_batch[:i]
                self.save_batch = self.save_batch[i:] + batch
                return default_collate(out_batch)

        out_batch = self.save_batch

        for i in range(1, len(batch)):
            if batch[i][1].size != size:
                self.save_batch = batch[i:]
                return default_collate(out_batch + batch[:i])
        
        out_batch += batch
        self.save_batch = None
        return default_collate(out_batch)
        
class ActionSchemaIterableDataset(IterableDataset):
    def __init__(self, feature_generator: CostPartitionFeatures, dataset: Dataset, action_schema: str):
        super().__init__()
        self.fg = feature_generator
        self.data = dataset
        self.use_cache = False
        self.action_schema = action_schema
        self.cache = []

    def __iter__(self):
        # worker_info = get_worker_info()
        
        # if worker_info is None:
        #     num_worker = 1
        # else:
        #     num_worker = worker_info.num_workers
        #     per_worker = int(math.ceil(len(self.data.y) / float(num_worker)))
        #     worker_id = worker_info.id
        #     worker_start = worker_id * per_worker
        #     worker_end = min(worker_start + per_worker, len(self.data.y))

        if not self.use_cache:
            for i, input in enumerate(self.fg.actions_embed_dataset(self.data.wlplan_dataset)):
                # if num_worker > 1:
                #     if i < worker_start or i >= worker_end:
                #         continue
                
                for action_name in input:
                    action_schema = get_action_schema_name(action_name)
                    if (action_schema == self.action_schema):
                        X = torch.from_numpy(np.array(input[action_name]))
                        y = torch.from_numpy(np.array(self.data.y[i][action_name]))

                        self.cache.append((X,y))
                        yield X,y
            self.use_cache = True
        else:
            random.shuffle(self.cache)
            for i, train_data in enumerate(self.cache):
                # if num_worker > 1:
                #     if i < worker_start or i >= worker_end:
                #         continue
                
                yield train_data


def embed_data(dataset: Dataset, feature_generator: Features, opts: Namespace):
    if opts.data_pruning == "none":
        X = feature_generator.embed(dataset.wlplan_dataset)
        X = np.array(X).astype(float)
        y = dataset.y
        sample_weight = None
    elif opts.data_pruning == "equivalent-weighted":
        X, y, sample_weight = get_data_weighted(dataset, feature_generator, opts)
    elif opts.data_pruning == "equivalent":
        X, y, _ = get_data_weighted(dataset, feature_generator, opts)
        sample_weight = None
    else:
        raise ValueError(f"Unknown data pruning method: {opts.data_pruning}")
    return X, y, sample_weight


def get_action_schemas_data(dataset: Dataset, feature_generator: Features) -> ActionSchemaIterableDataset:
    assert isinstance(dataset, CostPartitionDataset)
    assert isinstance(feature_generator, CostPartitionFeatures)

    return [ActionSchemaIterableDataset(feature_generator, dataset, action_schema.name) for action_schema in dataset.domain.action_schemas]


def get_action_schema_name(name: str):
    return name.split(" ")[0]

def get_data_weighted(dataset: Dataset, feature_generator: Features, opts: Namespace):
    if opts.rank:
        assert isinstance(dataset, RankingDataset)
        dataset: RankingDataset = dataset
        unique_groups = {}
        sample_weight_dict = {}
        counter = 0
        graphs = feature_generator.convert_to_graphs(dataset.wlplan_dataset)
        for ranking_group in tqdm(dataset.y, total=len(dataset.y)):
            gg = ranking_group.good_group
            mg = ranking_group.maybe_group
            bg = ranking_group.bad_group
            good_graphs = [graphs[i] for i in gg]
            maybe_graphs = [graphs[i] for i in mg]
            bad_graphs = [graphs[i] for i in bg]
            good_x = feature_generator.embed(good_graphs)
            maybe_x = feature_generator.embed(maybe_graphs)
            bad_x = feature_generator.embed(bad_graphs)
            good_x = sorted(tuple(x) for x in good_x)
            maybe_x = sorted(tuple(x) for x in maybe_x)
            bad_x = sorted(tuple(x) for x in bad_x)
            key = (tuple(good_x), tuple(maybe_x), tuple(bad_x))
            if key not in unique_groups:
                ranking_group.good_group = [i + counter for i in range(len(gg))]
                counter += len(gg)
                ranking_group.maybe_group = [i + counter for i in range(len(mg))]
                counter += len(mg)
                ranking_group.bad_group = [i + counter for i in range(len(bg))]
                counter += len(bg)
                unique_groups[key] = ranking_group
                sample_weight_dict[key] = 0
            sample_weight_dict[key] += 1
        X = []
        y = []
        sample_weight = []
        for key, ranking_group in unique_groups.items():
            for i in range(3):
                for x in key[i]:
                    X.append(x)
            y.append(ranking_group)
            sample_weight.append(sample_weight_dict[key])
        X = np.array(X).astype(float)
        return X, y, sample_weight
    else:
        unique_rows = {}
        graphs = feature_generator.convert_to_graphs(dataset.wlplan_dataset)
        for graph, y in tqdm(zip(graphs, dataset.y), total=len(graphs)):
            x = feature_generator.embed(graph)
            xy = np.array(x + [y])
            xy = tuple(xy)
            if xy not in unique_rows:
                unique_rows[xy] = 0
            unique_rows[xy] += 1
        Xy = []
        sample_weight = []
        for xy, count in unique_rows.items():
            Xy.append(xy)
            sample_weight.append(count)
        Xy = np.array(Xy).astype(float)
        sample_weight = np.array(sample_weight).astype(float)
        assert Xy.shape[0] == sample_weight.shape[0]
        return Xy[:, :-1], Xy[:, -1], sample_weight
