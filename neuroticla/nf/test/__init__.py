from transformers import TrainingArguments

from .common import run_test
from ...core.labels import BinaryLabeler, MultiLabeler, Labeler
from ...core.results import ResultWriter
from ...core.split import DataSplit
from ...core.trans import SeqClassifyModel
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
        help='Target model name. (overrides other settings)',
    )
    labels = ','.join(get_all_labels(module_name))
    parser.add_argument(
        '-u', '--subset', type=str, default=None, required=False,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    parser.add_argument('pretrained_model', default=None,
                        help='Pretrained model used for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])

    parser.add_argument('corpora', type=str, default=None,
                        help='Corpora prefix or path prefix to use for training')


def _get_training_args(arg, result_path: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=result_path,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        disable_tqdm=not arg.tqdm
    )


def test_binrel(arg) -> int:
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    text_fields = ['body']
    compute_m_name = arg.model_name is None

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        # model name w/o label
        arg.model_name = compute_model_name(arg, text_fields, None, True)
        result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started testing model [%s] for label [%s] from path [%s].',
                    arg.model_name, label, result_path)

        logger.info('Training for label: %s with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            result_path,
            labeler=BinaryLabeler(labels=[label]),
            device=arg.device
        )
        collector = ResultsCollector()
        results = run_test(
            arg, mc, _get_training_args(arg, result_path), test_data, label, text_fields, collector.collect
        )
        result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
        result_writer.write(results, 'binrel-' + arg.model_name + '-' + label, label)
        test_data['p_' + label] = collector.y_pred

        # reset model name back to None to be recomputed
        if compute_m_name:
            arg.model_name = None

    arg.model_name = compute_model_name(arg, text_fields, None, True)  # model name w/o label
    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(compute_model_path(arg, 'binrel'), arg.model_name + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)
    return 0


def test_lpset(arg) -> int:
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    text_fields = ['body']
    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))

    arg.model_name = compute_model_name(arg, text_fields, labels)
    result_path = compute_model_path(arg, 'lpset')
    logger.info('Started testing model [%s] for label [%s] from path [%s].',
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
    result_writer.write(results, 'lpset-' + arg.model_name)

    for lx, lbl in enumerate(labels):
        test_data['p_' + lbl] = [item[lx] for item in collector.y_pred]

    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(compute_model_path(arg, 'lpset'), arg.model_name + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)
    return 0
