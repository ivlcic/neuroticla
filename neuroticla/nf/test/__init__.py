import random
import pandas as pd
import numpy as np

from typing import Any

from transformers import TrainingArguments

from .common import run_test
from ...core.labels import BinaryLabeler, MultiLabeler, Labeler
from ...core.results import ResultWriter
from ...core.split import DataSplit
from ...core.trans import SeqClassifyModel, ClassificationMetrics
from ...nf.utils import *

logger = logging.getLogger('nf.test')


class ResultsCollector:

    def __init__(self):
        self.labeler = None
        self.y_pred = None
        self.y_true = None

    def collect(self, labeler: Labeler, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labeler = labeler


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(module_name, parser)
    parser.add_argument(
        '-n', '--model_name', type=str, default=None,
        help='Target model name. (overrides other settings used for model name construction)',
    )
    labels = ','.join(get_all_labels())
    parser.add_argument(
        '-u', '--subset', type=str, default=None, required=False,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    text_fields = ','.join(get_all_text_fields())
    parser.add_argument(
        '-f', '--fields', type=str, default='body', required=False,
        help='Text fields to use for testing: ' + text_fields + ')',
    )
    parser.add_argument(
        '-p', '--pretrained_model', type=str, default=None, required=False,
        help='Pretrained model that was used for fine tuning (used only for model name construction)',
        choices=['mcbert', 'xlmrb', 'xlmrl']
    )

    parser.add_argument('corpora', type=str, default=None,
                        help='Corpora prefix or path prefix to use for training')


def _get_training_args(arg, result_path: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=result_path,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        disable_tqdm=not arg.tqdm
    )


def test_majority(arg) -> int:
    labels = get_labels(arg)
    if not labels:
        return 1

    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    arg.model_name = f'majority.{arg.corpora}{get_labels_str(labels)}'
    result_path = compute_model_path(arg, 'baseline')
    logger.info('Started testing model [%s] for labels %s from path [%s].',
                arg.model_name, labels, result_path)

    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    global_true = []
    global_pred = []
    for label in labels:
        y_pred = [0] * test_data.shape[0]  # all zeros is a majority class / label combination
        test_data['p_' + label] = y_pred
        global_pred.append(y_pred)
        global_true.append(test_data[label].tolist())

    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(os.path.dirname(result_path), arg.model_name + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)

    # write label indicator arrays
    global_true: List[Any] = np.array(global_true).transpose().tolist()
    global_pred: List[Any] = np.array(global_pred).transpose().tolist()
    metrics = ClassificationMetrics()
    global_results = metrics.compute(references=global_true, predictions=global_pred, labels=labels)

    result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
    result_writer.write(global_results, 'baseline-' + arg.model_name)
    return 0


def test_random(arg) -> int:
    labels = get_labels(arg)
    if not labels:
        return 1

    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    arg.model_name = f'random.{arg.corpora}{get_labels_str(labels)}'
    result_path = compute_model_path(arg, 'baseline')
    logger.info('Started testing model [%s] for labels %s from path [%s].',
                arg.model_name, labels, result_path)

    # create random test set
    data = pd.concat([train_data, eval_data, test_data], ignore_index=True)
    # Create a new column to store the label combinations
    data['comb'] = data[labels].apply(lambda row: ''.join(row.astype(str)), axis=1)
    # Calculate frequencies for each string combination
    combination_freq = data['comb'].value_counts().reset_index()
    combination_freq.columns = ['comb', 'freq']

    combos = pd.DataFrame({
        'comb': random.choices(
                    combination_freq['comb'].values.tolist(),
                    combination_freq['freq'].values.tolist(),
                    k=test_data.shape[0]
                )
    })
    split_combos = combos['comb'].apply(lambda x: pd.Series([int(i) for i in list(x)]))
    split_combos.columns = labels

    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    global_true = []
    global_pred = []
    for label in labels:
        y_pred = split_combos[label].values.tolist()
        test_data['p_' + label] = y_pred
        global_pred.append(y_pred)
        global_true.append(test_data[label].tolist())

    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(os.path.dirname(result_path), arg.model_name + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)

    # write label indicator arrays
    global_true: List[Any] = np.array(global_true).transpose().tolist()
    global_pred: List[Any] = np.array(global_pred).transpose().tolist()
    metrics = ClassificationMetrics()
    global_results = metrics.compute(references=global_true, predictions=global_pred, labels=labels)

    result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
    result_writer.write(global_results, 'baseline-' + arg.model_name)
    return 0


def test_binrel(arg) -> int:
    labels = get_labels(arg)
    if not labels:
        return 1

    l_str = get_labels_str(labels)
    text_fields = get_text_fields(arg)
    logger.info('Started testing for labels %s and text fields %s.', labels, text_fields)

    computed_name = arg.model_name

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    global_true = []
    global_pred = []
    for label in labels:
        # model name w/o label
        arg.model_name = compute_model_name(arg, text_fields, None, True)
        result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started testing model [%s] for label [%s] from path [%s].',
                    arg.model_name, label, result_path)

        logger.info('Testing for label: %s with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            result_path,
            labeler=BinaryLabeler(labels=[label]),
            device=arg.device
        )
        collector = ResultsCollector()
        results = run_test(
            arg, mc, _get_training_args(arg, result_path), test_data, label, text_fields, collector.collect
        )
        result_writer = ResultWriter(
            arg.result_dir, os.path.dirname(result_path), None
        )
        result_writer.write(results, 'binrel.' + arg.model_name + get_labels_str([label]), label)
        test_data['p_' + label] = collector.y_pred
        global_true.append(collector.y_true)
        global_pred.append(collector.y_pred)

        # reset model name back to original
        if computed_name:
            arg.model_name = computed_name

    # write predictions
    arg.model_name = compute_model_name(arg, text_fields, None, True)  # model name w/o label
    result_path = os.path.join(compute_model_path(arg, 'binrel'))
    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(os.path.dirname(result_path), 'binrel.' + arg.model_name + l_str + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)

    # compute combined results
    global_true: List[Any] = np.array(global_true).transpose().tolist()
    global_pred: List[Any] = np.array(global_pred).transpose().tolist()
    metrics = ClassificationMetrics()
    global_results = metrics.compute(references=global_true, predictions=global_pred, labels=labels)

    result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
    result_writer.write(global_results, 'binrel.' + arg.model_name + l_str)
    return 0


def test_lpset(arg) -> int:
    labels = get_labels(arg)
    if not labels:
        return 1

    text_fields = get_text_fields(arg)
    logger.info('Started testing for labels %s and text fields %s.', labels, text_fields)

    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))

    arg.model_name = compute_model_name(arg, text_fields, labels)
    result_path = compute_model_path(arg, 'lpset')
    logger.info('Started testing model [%s] for labels %s from path [%s].',
                arg.model_name, labels, result_path)

    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    mc = SeqClassifyModel(
        result_path,
        labeler=MultiLabeler(labels=labels),
        device=arg.device
    )

    collector = ResultsCollector()
    results = run_test(
        arg, mc, _get_training_args(arg, result_path), test_data, labels, text_fields, collector.collect
    )
    result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
    result_writer.write(results, 'lpset.' + arg.model_name)

    for lx, lbl in enumerate(labels):
        test_data['p_' + lbl] = [item[lx] for item in collector.y_pred]

    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(os.path.dirname(result_path), 'lpset.' + arg.model_name + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)
    return 0
