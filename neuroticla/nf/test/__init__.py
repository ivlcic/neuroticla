from transformers import TrainingArguments

from .common import run_test, write_test_results
from ...core.labels import BinaryLabeler, MultiLabeler
from ...core.split import DataSplit
from ...core.trans import SeqClassifyModel, ModelContainer
from ...nf.utils import *

logger = logging.getLogger('nf.test')


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
    logger = logging.getLogger('nf.test.binrel')
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    compute_m_name = arg.model_name is None

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        arg.model_name = compute_model_name(arg, None, True)  # model name w/o label
        result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started testing model [%s] for label [%s] from path [%s].',
                    arg.model_name, label, result_path)

        logger.info('Training for label: %s with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            result_path,
            labeler=BinaryLabeler(labels=[label]),
            device=arg.device
        )

        results = run_test(arg, mc, _get_training_args(arg, result_path), test_data, label)
        write_test_results(arg, results, [label])

        # reset model name back to None to be recomputed
        if compute_m_name:
            arg.model_name = None

    return 0


def test_lpset(arg) -> int:
    logger = logging.getLogger('nf.test.lpset')
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))

    compute_model_name(arg, labels)
    result_path = compute_model_path(arg, 'lpset')
    logger.info('Started testing model [%s] for label [%s] from path [%s].',
                arg.model_name, labels, result_path)

    logger.info('Testing labels: %s with device [%s]', labels, arg.device)
    mc = SeqClassifyModel(
        result_path,
        labeler=MultiLabeler(labels=labels),
        device=arg.device
    )

    results = run_test(arg, mc, _get_training_args(arg, result_path), test_data, labels)
    write_test_results(arg, results, labels)
    return 0
