import os.path
from typing import Any

import numpy as np

from transformers import TrainingArguments

from ..test import run_test, ResultsCollector
from ...core.dataset import SeqClassifyDataset
from ...core.labels import BinaryLabeler, MultiLabeler
from ...core.results import ResultWriter
from ...core.split import DataSplit
from ...core.trans import SeqClassifyModel, ModelContainer
from ...core.eval import MultilabelMetrics
from ...nf.utils import *

logger = logging.getLogger('nf.train')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(module_name, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    labels = ','.join(get_all_labels())
    parser.add_argument(
        '-u', '--subset', type=str, default=None,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    text_fields = ','.join(get_all_text_fields())
    parser.add_argument(
        '-f', '--fields', type=str, default='body', required=False,
        help='Text fields to use for training: ' + text_fields + ')',
    )
    parser.add_argument(
        '-p', '--pretrained_model', type=str, default=None, required=False,
        help='Pretrained model that should be used for fine tuning',
        choices=['mcbert', 'xlmrb', 'xlmrl']
    )
    parser.add_argument(
        '-m', '--metric', type=str, default='macro', required=False,
        help='Metric to select for best model selection (used only for model name construction)',
        choices=['micro-1', 'micro', 'micro-1', 'macro']
    )
    parser.add_argument('corpora', type=str, default=None,
                        help='Corpora prefix or path prefix to use for training')


def _get_training_args(arg, result_path: str) -> TrainingArguments:
    return TrainingArguments(
            output_dir=result_path,
            num_train_epochs=arg.epochs,
            per_device_train_batch_size=arg.batch,
            per_device_eval_batch_size=arg.batch,
            evaluation_strategy='epoch',
            disable_tqdm=not arg.tqdm,
            load_best_model_at_end=True,
            save_strategy='epoch',
            learning_rate=arg.learn_rate,
            optim='adamw_torch',
            # optim='adamw_hf',
            save_total_limit=1,
            metric_for_best_model='f1',
            logging_strategy='epoch',
        )


def train_binrel(arg) -> int:
    logger.info('Starting binary relevance training ...')
    labels = get_labels(arg)
    if not labels:
        return 1
    l_str = get_labels_str(labels)
    text_fields = get_text_fields(arg)
    logger.info('Started training for labels %s and text fields %s.', labels, text_fields)
    computed_name = arg.model_name

    global_true = []
    global_pred = []
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        arg.model_name = compute_model_name(arg, text_fields, None, True)  # model name w/o label
        result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started training model [%s] for label [%s] to path [%s].',
                    arg.model_name, label, result_path)

        logger.info('Training for label: %s with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            ModelContainer.model_name_map[arg.pretrained_model],
            labeler=BinaryLabeler(labels=[label]),
            cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
            device=arg.device,
            best_metric=arg.metric
        )
        logger.debug('Constructing train data set [%s]...', len(train_data))
        train_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, label, text_fields
        )
        logger.info('Constructed train data set [%s].', len(train_data))
        logger.debug('Constructing evaluation data set [%s]...', len(eval_data))
        eval_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, label, text_fields
        )
        logger.info('Constructed evaluation data set [%s].', len(eval_data))

        training_args = _get_training_args(arg, result_path)
        # train the model
        mc.build(training_args, train_set, eval_set)

        # test the model
        collector = ResultsCollector()
        results = run_test(
            arg, mc, training_args, test_data, label, text_fields, collector.collect
        )
        result_writer = ResultWriter(
            arg.result_dir, os.path.dirname(result_path), None
        )
        result_writer.write(results, 'binrel.' + arg.model_name + get_labels_str([label]), label)
        test_data['p_' + label] = collector.y_pred
        global_true.append(collector.y_true)
        global_pred.append(collector.y_pred)

        ModelContainer.remove_checkpoint_dir(result_path)

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
    metrics = MultilabelMetrics()
    global_results = metrics.compute(references=global_true, predictions=global_pred, labels=labels)

    result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
    result_writer.write(global_results, 'binrel.' + arg.model_name + l_str)
    return 0


def train_lpset(arg) -> int:
    logger.info("Starting label power-set training ...")
    labels = get_labels(arg)
    if not labels:
        return 1

    text_fields = get_text_fields(arg)
    logger.info('Started training for labels %s and text fields %s.', labels, text_fields)
    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))

    arg.model_name = compute_model_name(arg, text_fields, labels)
    result_path = compute_model_path(arg, 'lpset')

    logger.info('Training for labels: %s with device [%s]', labels, arg.device)
    mc = SeqClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        labeler=MultiLabeler(labels=labels),
        cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
        device=arg.device,
        best_metric=arg.metric
    )

    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, labels, text_fields
    )
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, labels, text_fields
    )
    logger.info("Constructed evaluation data set [%s].", len(eval_data))

    training_args = _get_training_args(arg, result_path)
    # train the model
    mc.build(training_args, train_set, eval_set)

    collector = ResultsCollector()
    results = run_test(
        arg, mc, training_args, test_data, labels, text_fields, collector.collect
    )
    result_writer = ResultWriter(arg.result_dir, os.path.dirname(result_path))
    result_writer.write(results, 'lpset.' + arg.model_name)

    for lx, lbl in enumerate(labels):
        test_data['p_' + lbl] = [item[lx] for item in collector.y_pred]

    test_data.drop(['body', 'lead'], axis=1, inplace=True)
    test_pred_path = os.path.join(os.path.dirname(result_path), 'lpset.' + arg.model_name + '.cvs')
    test_data.to_csv(test_pred_path, encoding='utf-8', index=False)

    ModelContainer.remove_checkpoint_dir(result_path)
    return 0
