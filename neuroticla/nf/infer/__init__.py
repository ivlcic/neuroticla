import numpy as np

from .common import (_get_train_params, _get_inference_data, _infer, _get_training_args,
                     _write_results, _get_baseline_params)
from ...core.labels import BinaryLabeler, MultiLabeler
from ...core.results import ResultsCollector
from ...core.trans import SeqClassifyModel
from ...nf.utils import *

logger = logging.getLogger('nf.infer')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.test(parser, 24)
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-m', '--model_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
    parser.add_argument(
        '-n', '--model_name', type=str, default=None,
        help='Trained model name or path.',
    )
    text_fields = ','.join(get_all_text_fields())
    parser.add_argument(
        '-f', '--fields', type=str, default=None, required=False,
        help='Text fields to use for inference: ' + text_fields + ')',
    )
    parser.add_argument('input_file', type=str, default=None,
                        help='File prefix or path prefix to use for inference')


def infer_majority_0(arg) -> int:
    labels, data, _ = _get_baseline_params(arg)

    model_name = f'majority-0.{arg.input_file}{get_labels_str(labels)}'
    logger.info('Started model [%s] inference for labels %s.', model_name, labels)
    collector = ResultsCollector()
    for label in labels:
        y_pred = [0] * data.shape[0]  # all zeros are the majority class / label set
        data['p_' + label] = y_pred
        collector.collect(None, data[label].tolist(), y_pred)
    _write_results(arg, model_name, data, collector.get_all_true(), collector.get_all_pred(), labels)
    return 0


def infer_majority_l(arg) -> int:
    labels, data, _ = _get_baseline_params(arg)
    model_name = f'majority-l.{arg.input_file}{get_labels_str(labels)}'
    logger.info(
        'Started majority non-zero label set model [%s] inference for labels %s.', model_name, labels
    )
    collector = ResultsCollector()
    for label in labels:
        if label == 'sec':
            y_pred = [1] * data.shape[0]  # security is of majority class / label
        else:
            y_pred = [0] * data.shape[0]  # all other are zeros
        data['p_' + label] = y_pred
        collector.collect(None, data[label].tolist(), y_pred)

    _write_results(arg, model_name, data, collector.get_all_true(), collector.get_all_pred(), labels)
    return 0


def infer_random(arg) -> int:
    labels, data, random_data = _get_baseline_params(arg)
    model_name = f'random.{arg.input_file}{get_labels_str(labels)}'
    logger.info('Started random model [%s] inference for labels %s.', model_name, labels)
    collector = ResultsCollector()
    for label in labels:
        y_pred = random_data[label].values.tolist()
        data['p_' + label] = y_pred
        collector.collect(None, data[label].tolist(), y_pred)

    _write_results(arg, model_name, data, collector.get_all_true(), collector.get_all_pred(), labels)
    return 0


def infer_binrel(arg) -> int:
    train_params, model_dir, model_name = _get_train_params(arg)

    labels = train_params['labels'].split(',')
    if arg.fields is not None:
        text_fields = get_text_fields(arg.fields)
    else:
        text_fields = train_params['train_fields'].split(',')

    logger.info(
        'Started model %s inference for labels %s and text fields %s.',
        train_params['name'], labels, text_fields
    )
    data = _get_inference_data(arg, text_fields)

    collector = ResultsCollector()
    for label in labels:
        sub_model_dir = os.path.join(model_dir, label)
        logger.info('Started predicting with model [%s] for label [%s] from path [%s].',
                    arg.model_name, label, sub_model_dir)

        logger.info('Predicting label: [%s] with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            sub_model_dir,
            labeler=BinaryLabeler(labels=[label]),
            device=arg.device
        )
        predictions, true_values = _infer(
            arg, mc, _get_training_args(arg, sub_model_dir), data, text_fields
        )
        data['p_' + label] = predictions
        if true_values is not None:
            collector.collect(mc.labeler(), true_values, predictions)
        mc.destroy()

    _write_results(arg, model_name, data, collector.get_all_true(), collector.get_all_pred(), labels)

    return 0


def infer_lpset(arg) -> int:
    train_params, model_dir, model_name = _get_train_params(arg)

    labels = train_params['labels'].split(',')
    if arg.fields is not None:
        text_fields = get_text_fields(arg.fields)
    else:
        text_fields = train_params['train_fields'].split(',')

    logger.info(
        'Started model %s inference for labels %s and text fields %s.',
        train_params['name'], labels, text_fields
    )
    data = _get_inference_data(arg, text_fields)
    mc = SeqClassifyModel(
        model_dir,
        labeler=MultiLabeler(labels=labels),
        device=arg.device
    )
    predictions, true_values = _infer(
        arg, mc, _get_training_args(arg, model_dir), data, text_fields
    )
    y_pred = np.transpose(predictions)
    for idx, label in enumerate(labels):
        data['p_' + label] = y_pred[idx]

    _write_results(arg, model_name, data, true_values, predictions, labels)
    return 0
