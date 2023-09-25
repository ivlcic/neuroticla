import random
import socket
import pandas as pd

from typing import Tuple

from transformers import TrainingArguments

from ..core.dataset import SeqClassifyDataset
from ..core.eval import MultilabelMetrics
from ..core.labels import BinaryLabeler, MultiLabeler, Labeler
from ..core.results import ResultWriter
from ..core.results import ResultsCollector
from ..core.split import DataSplit
from ..core.trans import SeqClassifyModel
from ..nf.utils import *

logger = logging.getLogger('nf.test')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
    parser.add_argument(
        '-n', '--model_name', type=str, default=None,
        help='Target model name. (overrides other settings used for model name construction)',
    )
    text_fields = ','.join(get_all_text_fields())
    parser.add_argument(
        '-f', '--test_fields', type=str, default=None, required=False,
        help=f'Text fields to use for testing: ' + text_fields + '. If not specified train fields are used.',
    )
    parser.add_argument(
        '-x', '--eval_source', type=str, default='', required=False,
        help='Evaluate model on specified file else on corpora test split',
    )


def _get_training_args(arg, result_path: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=result_path,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        disable_tqdm=not arg.tqdm
    )


def _load_test_data(arg) -> Tuple[pd.DataFrame, str]:
    input_file = os.path.join(arg.data_in_dir, arg.eval_source)
    if not os.path.exists(input_file):
        input_file = arg.eval_source

    if not os.path.exists(input_file):
        if arg.corpora is None:
            arg.corpora = arg.eval_source
        base_name = ''
        _, _, test_data = DataSplit.load(get_data_path_prefix(arg))
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0] + '.'
        test_data = pd.read_csv(input_file, encoding='utf-8')
    return test_data, base_name


def _write_metrics_results(arg, labels: List[str], result_path: str,
                           collector: ResultsCollector, test_data: pd.DataFrame):
    metrics = MultilabelMetrics()
    results = metrics.compute(
        references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
    )
    result_writer = ResultWriter()
    # write model results
    result_writer.write_predictions(result_path, 'predictions', test_data, ['body', 'lead'])
    result_writer.write_metrics(result_path, 'metrics', arg.model_name, results, True)
    # write global results
    result_writer.write_metrics(arg.result_dir, 'results_' + socket.gethostname(), arg.model_name, results)


def test_majority0(arg) -> int:
    labels = get_labels(arg)
    test_data, base_name = _load_test_data(arg)

    arg.model_name = f'majority-0.{arg.corpora}{get_labels_str(labels)}'
    result_path = compute_model_path(arg, 'baseline')
    logger.info('Started testing model [%s] for labels %s from path [%s].',
                arg.model_name, labels, result_path)

    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    collector = ResultsCollector()
    for label in labels:
        y_pred = [0] * test_data.shape[0]  # all zeros are the majority class / label set
        test_data['p_' + label] = y_pred
        collector.collect(None, test_data[label].tolist(), y_pred)
    _write_metrics_results(arg, labels, result_path, collector, test_data)
    return 0


def test_majority_labeled(arg) -> int:
    labels = get_labels(arg)
    test_data, base_name = _load_test_data(arg)
    arg.model_name = f'majority-l.{arg.corpora}{get_labels_str(labels)}'
    result_path = compute_model_path(arg, 'baseline')
    logger.info('Started testing model [%s] for labels %s from path [%s].',
                arg.model_name, labels, result_path)

    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    collector = ResultsCollector()
    for label in labels:
        if label == 'sec':
            y_pred = [1] * test_data.shape[0]  # security is of majority class / label
        else:
            y_pred = [0] * test_data.shape[0]  # all other are zeros
        test_data['p_' + label] = y_pred
        collector.collect(None, test_data[label].tolist(), y_pred)

    _write_metrics_results(arg, labels, result_path, collector, test_data)
    return 0


def _random_freq_data(arg, labels) -> pd.DataFrame:
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
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
    return split_combos


def test_random(arg) -> int:
    labels = get_labels(arg)
    test_data, base_name = _load_test_data(arg)
    arg.model_name = f'random.{arg.corpora}{get_labels_str(labels)}'
    result_path = compute_model_path(arg, 'baseline')
    logger.info('Started testing model [%s] for labels %s from path [%s].',
                arg.model_name, labels, result_path)

    random_data = _random_freq_data(arg, labels)
    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    collector = ResultsCollector()
    for label in labels:
        y_pred = random_data[label].values.tolist()
        test_data['p_' + label] = y_pred
        collector.collect(None, test_data[label].tolist(), y_pred)

    _write_metrics_results(arg, labels, result_path, collector, test_data)
    return 0


def _test_model(arg, labeler: Labeler, result_path: str, collector: ResultsCollector, test_data: pd.DataFrame):
    test_fields = get_test_fields(arg)
    labels = labeler.source_labels()
    logger.info('Testing labels: [%s] with device [%s]', labels, arg.device)
    mc = SeqClassifyModel(
        result_path,
        labeler=labeler,
        device=arg.device
    )

    # run tests
    logger.debug('Constructing test data set [%s]...', len(test_data))
    test_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, labels, test_fields
    )
    logger.info('Constructed test data set [%s].', len(test_data))
    results = mc.test(_get_training_args(arg, result_path), test_set, collector.collect)
    logger.info('Test set evaluation results: [%s].', results)

    if isinstance(labels, str):
        test_data['p_' + labels] = collector.y_pred
    else:
        for lx, lbl in enumerate(labels):
            test_data['p_' + lbl] = [item[lx] for item in collector.y_pred]

    return results


def test_binrel(arg) -> int:
    result_path = os.path.join(compute_model_path(arg, 'binrel'))
    read_model_params(result_path, arg)
    test_data, base_name = _load_test_data(arg)

    labels = get_labels(arg)
    collector = ResultsCollector()
    for label in labels:
        sub_result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started training model [%s] for label [%s] to path [%s].',
                    arg.model_name, label, sub_result_path)
        _test_model(
            arg, BinaryLabeler(labels=[label]), sub_result_path, collector, test_data
        )

    metrics = MultilabelMetrics()
    results = metrics.compute(
        references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
    )

    result_writer = ResultWriter()
    # write model results
    result_writer.write_predictions(result_path, base_name + 'predictions', test_data, ['body', 'lead'])
    result_writer.write_metrics(result_path, base_name + 'metrics', arg.model_name, results, True)
    # write global results
    result_writer.write_metrics(
        arg.result_dir, 'results_' + socket.gethostname(), base_name + arg.model_name, results
    )
    return 0


def test_lpset(arg) -> int:
    result_path = os.path.join(compute_model_path(arg, 'lpset'))
    read_model_params(result_path, arg)
    test_data, base_name = _load_test_data(arg)

    labels = get_labels(arg)
    collector = ResultsCollector()
    results = _test_model(arg, MultiLabeler(labels=labels), result_path, collector, test_data)

    result_writer = ResultWriter()
    # write model results
    result_writer.write_predictions(result_path, base_name + 'predictions', test_data, ['body', 'lead'])
    result_writer.write_metrics(result_path, base_name + 'metrics', arg, results, True)
    # write global results
    result_writer.write_metrics(
        arg.result_dir, 'results_' + socket.gethostname(), base_name + arg.model_name, results
    )
    return 0
