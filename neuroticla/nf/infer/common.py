import os
import logging
import json
import pandas as pd

from typing import Union, List, Any, Tuple, Dict
from transformers import TrainingArguments

from ...core.dataset import SeqClassifyDataset, SeqEvalDataset
from ...core.eval import MultilabelMetrics
from ...core.results import ResultWriter
from ...core.trans import SeqClassifyModel

logger = logging.getLogger('nf.infer')


def _get_train_params(arg) -> Tuple[Dict[str, Any], str, str]:
    if not arg.model_name:
        raise ValueError(f'Missing model name param')
    model_train_file = os.path.join(arg.model_dir, arg.model_name, 'train_params.json')
    if not os.path.exists(model_train_file):
        model_train_file = os.path.join(arg.model_name, 'train_params.json')
    if not os.path.exists(model_train_file):
        raise ValueError(f'Missing model train params [{model_train_file}]')

    with open(model_train_file, 'r') as json_file:
        train_params = json.load(json_file)
    model_dir = os.path.dirname(model_train_file)
    model_name = os.path.basename(model_dir)
    return train_params, model_dir, model_name


def _get_training_args(arg, model_path: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        disable_tqdm=not arg.tqdm
    )


def _get_inference_data(arg, text_fields):
    input_file = os.path.join(arg.data_in_dir, arg.input_file)
    if not os.path.exists(input_file):
        input_file = arg.input_file
    if not os.path.exists(input_file):
        raise ValueError(f'Missing input file [{input_file}]')

    data: pd.DataFrame = pd.read_csv(
        input_file,
        encoding='utf-8',
        # nrows=128
    )
    for text_field in text_fields:
        if text_field not in data:
            raise ValueError(f'Missing text field [{text_field}] in input data [{input_file}]')
    return data


def _infer(arg, mc: SeqClassifyModel, infer_args: TrainingArguments, data: pd.DataFrame,
           text_field: Union[str, List[str]] = 'body'):

    labels = mc.labeler().source_labels()
    if isinstance(labels, str):
        labels = [labels]

    dataset_contains_labels = True  # we will compute evaluation
    for label in labels:
        if label not in data:
            dataset_contains_labels = False

    if dataset_contains_labels:
        logger.debug('Constructing inference data set [%s]...', data.shape[0])
        data_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), data, mc.max_len(), labels, text_field
        )
        logger.info('Constructed inference data set [%s].', data.shape[0])
    else:
        logger.debug('Constructing inference data set [%s]...', data.shape[0])
        data_set = SeqEvalDataset(
            mc.labeler(), mc.tokenizer(), data, mc.max_len(), text_field
        )
        logger.info('Constructed inference data set [%s].', data.shape[0])

    predictions, true_values = mc.infer_data_set(infer_args, data_set)
    if true_values is None:
        logger.info(
            'Inferred data set predictions [%s].', len(predictions)
        )
    else:
        logger.info(
            'Inferred data set predictions [%s] and true values [%s].', len(predictions), len(true_values)
        )
    return predictions, true_values


def _write_results(arg, model_name, data, true_values, predictions, labels):
    result_writer = ResultWriter()
    # write predictions
    base_name = os.path.splitext(os.path.basename(arg.input_file))[0]
    input_name = base_name + '.' + model_name
    result_writer.write_predictions(
        arg.result_dir, f'{input_name}.predictions', data, []
    )
    if true_values is not None and len(true_values) > 0:
        metrics = MultilabelMetrics()
        result = metrics.compute(
            references=true_values, predictions=predictions, labels=labels
        )
        result_writer.write_metrics(
            arg.result_dir, f'{base_name}.metrics', input_name, result
        )
