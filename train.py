#!/usr/bin/env python

import logging
import termcolor as tc

import comet_ml  # noqa: F401
import toml
import torch

from learning.dataset.dataset_factory import get_train_dataset, get_validation_dataset
from learning.dataset.state_to_vec import embed_data
from learning.dataset.torch_utils import get_action_schemas_data, collate_variable_seq
from learning.options import parse_opts
from learning.predictor.predictor_factory import get_predictor, is_rank_predictor, get_cost_partition_predictor
from util.distinguish_test import distinguish
from util.logging import init_logger
from util.pca_visualise import visualise
from util.statistics import log_quartiles
from util.timer import TimerContextManager
from wlplan.feature_generation import get_feature_generator
from wlplan.planning import parse_domain


def train(opts):
    if opts.task == "heuristic":
        opts.rank = is_rank_predictor(opts.optimisation)
    else:
        opts.rank = False

    # Find PyTorch device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("Found device: " + tc.colored(device, "cyan"))
    
    # Parse dataset
    with TimerContextManager("parsing training data"):
        domain_pddl = toml.load(opts.data_config)["domain_pddl"]
        domain = parse_domain(domain_pddl)
        features = opts.features
        graph_representation = opts.graph_representation
        logging.info(f"{features=}")
        logging.info(f"{graph_representation=}")
        feature_generator = get_feature_generator(
            feature_algorithm=features,
            graph_representation=graph_representation,
            domain=domain,
            iterations=opts.iterations,
            task=opts.task
        )
        feature_generator.print_init_colours()
        dataset = get_train_dataset(opts, feature_generator)
        logging.info(f"{len(dataset)=}")

    # Collect colours
    with TimerContextManager("collecting colours"):
        feature_generator.collect(dataset.wlplan_dataset)
    logging.info("n_colours_per_layer:")
    for i, n_colours in enumerate(feature_generator.get_layer_to_n_colours()):
        logging.info(f"  {i}={n_colours}")
    if opts.collect_only:
        logging.info("Exiting after collecting colours.")
        exit(0)

    if opts.task == "heuristic":
        # Construct features
        with TimerContextManager("constructing features"):
            X, y, sample_weight = embed_data(
                dataset=dataset, feature_generator=feature_generator, opts=opts
            )
        
        if not opts.rank:
            log_quartiles(y)

        logging.info(f"{X.shape=}")

        # distinct_per_column_counts = {}
        # for column in X.T:
        #     column = set(column)
        #     size = len(column)
        #     if size not in distinct_per_column_counts:
        #         distinct_per_column_counts[size] = 0
        #     distinct_per_column_counts[size] += 1
        # for k in sorted(distinct_per_column_counts.keys()):
        #     print(k, distinct_per_column_counts[k])

        # colour_counts = {}
        # for column in X.T:
        #     summ = sum(column)
        #     if summ not in colour_counts:
        #         colour_counts[summ] = 0
        #     colour_counts[summ] += 1
        # for k in sorted(colour_counts.keys()):
        #     print(k, colour_counts[k])

        # breakpoint()

        # PCA visualisation
        pca_save_file = opts.visualise_pca
        if pca_save_file is not None:
            visualise(X, y, save_file=pca_save_file)
            return

        # distinguishability testing
        if opts.distinguish_test:
            distinguish(X, y)
            return

        # Train model
        predictor = get_predictor(opts.optimisation)
        predictor.fit_evaluate(X, y, sample_weight=sample_weight)

        # Save model
        if opts.save_file:
            with TimerContextManager("saving model"):
                feature_generator.set_weights(predictor.get_weights())
                feature_generator.save(opts.save_file)

    else:

        with TimerContextManager("parsing validation data"):
            validation_dataset = get_validation_dataset(opts, feature_generator)

        num_action_schemas = len(domain.action_schemas)
        action_schema_names = [a.name for a in domain.action_schemas]

        schema_predictors = [get_cost_partition_predictor(opts.optimisation, feature_generator.get_n_features(),
                                                          domain.name, domain.action_schemas[k].name, feature_generator.get_iterations()) 
                             for k in range(num_action_schemas)]

        train_datasets = get_action_schemas_data(dataset, feature_generator)
        validation_datasets = get_action_schemas_data(validation_dataset, feature_generator)
        
        for i in range(len(schema_predictors)):
            with TimerContextManager(f"training predictor for {action_schema_names[i]}"):

                train_data_loader = torch.utils.data.DataLoader(train_datasets[i], batch_size=16, pin_memory=True, collate_fn=collate_variable_seq)
                validation_data_loader = torch.utils.data.DataLoader(validation_datasets[i], batch_size=16, pin_memory=True, collate_fn=collate_variable_seq)
                schema_predictors[i].fit(train_data_loader, validation_data_loader)

                train_datasets[i].purge_cache()
                validation_datasets[i].purge_cache()

        if opts.save_file:
            with TimerContextManager("saving model"):
                pass

if __name__ == "__main__":
    init_logger()
    opts = parse_opts()
    train(opts)
