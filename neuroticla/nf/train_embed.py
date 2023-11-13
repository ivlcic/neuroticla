import os
import logging
import pandas as pd
import numpy as np

from typing import Tuple, List
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from .train import _write_results
from .train import _save_predictions
from ..core.args import CommonArguments
from .utils import get_all_labels, get_labels, get_all_embeddings_fields, compute_model_path, \
    write_model_params, get_data_path_prefix, compute_model_name
from ..core.eval import MultilabelMetrics
from ..core.labels import MultiLabeler, BinaryLabeler
from ..core.results import ResultsCollector
from ..core.split import DataSplit

logger = logging.getLogger('nf.train_embed')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    parser.add_argument(
        '-k', '--k_fold', help='Do K-fold cross-validation.', type=int, default=0
    )
    labels = ','.join(get_all_labels())
    parser.add_argument(
        '-u', '--subset', type=str, default=None,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    embeddings_fields = ','.join(get_all_embeddings_fields())
    parser.add_argument(
        '-f', '--train_fields', type=str, default='embed_oai_ada2', required=False,
        help='Embeddings field to use for training: ' + embeddings_fields + ')',
    )
    parser.add_argument(
        '-a', '--algorithm', type=str, default='rforest', required=False,
        help='Algorithm that should be used for classification.',
        choices=['rforest']
    )
    parser.add_argument(
        '-m', '--metric', type=str, default='micro-1', required=False,
        help='Metric to select for best model selection (used only for model name construction)',
        choices=['micro-1', 'micro', 'macro-1', 'macro']
    )
    parser.add_argument('corpora', type=str, default=None,
                        help='Corpora prefix or path prefix to use for training')


def _prep(arg, pt_method: str) -> Tuple[str, List[str]]:
    # compute a model name from the train params if not given
    arg.model_name = compute_model_name(arg, pt_method, nn=False)
    # compute final model collection path based on model name
    fold = 't'
    if arg.k_fold > 0:
        fold = arg.k_fold
    result_path = os.path.join(compute_model_path(arg.result_dir, f'k{fold}.' + arg.model_name))
    # store all input parameters
    params = write_model_params(result_path, arg, pt_method, nn=False)
    # determine which labels to use - all or just a subset
    labels = get_labels(arg)
    logger.info(
        'Starting training model [%s] for labels [%s] for params %s with result path [%s]...',
        arg.model_name, labels, params, result_path
    )
    return result_path, labels


def train_embed_binrel(arg) -> int:
    logger.info("Starting embeddings label binary relevance training ...")
    result_path, labels = _prep(arg, 'binrel')
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    test_data = pd.concat([eval_data, test_data])
    if arg.k_fold == 0:
        collector = ResultsCollector()
        train_data_embeddings = train_data[arg.train_fields].apply(np.array).to_list()
        test_data_embeddings = test_data[arg.train_fields].apply(np.array).to_list()
        for label in labels:
            sub_result_path = os.path.join(
                result_path, label
            )
            logger.info('Started training model [%s] for label [%s] to path [%s].',
                        arg.model_name, label, sub_result_path)

            y_train_true = train_data[label].to_list()
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_embeddings, y_train_true)

            y_test_true = test_data[label].to_list()
            y_test_pred = clf.predict(test_data_embeddings)
            collector.collect(BinaryLabeler(labels=[label]), y_test_true, y_test_pred)
            # write predictions to the test-set dataframe
            _save_predictions([label], test_data, collector)

        metrics = MultilabelMetrics()
        result = metrics.compute(
            references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
        )
        _write_results(arg, labels, result_path, result, test_data)
    return 0


def train_embed_lpset(arg) -> int:
    logger.info("Starting embeddings label power-set training ...")
    result_path, labels = _prep(arg, 'lpset')
    # do a classical split
    labeler: MultiLabeler = MultiLabeler(labels=labels)
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    test_data = pd.concat([eval_data, test_data])
    if arg.k_fold == 0:
        X_train = list(train_data[arg.train_fields].apply(np.array).values)
        y_train = labeler.encode_columns(train_data)
        X_test = list(test_data[arg.train_fields].apply(np.array).values)
        y_test = labeler.encode_columns(test_data)

        # train random forest classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        report = classification_report(y_test, preds)
        print(report)
    return 0
