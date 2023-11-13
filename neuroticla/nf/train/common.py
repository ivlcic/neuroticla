import os
import logging
import shutil
import socket
from typing import Tuple, List, Union, Dict, Any

import pandas as pd
from transformers import TrainingArguments

from ...core.dataset import SeqClassifyDataset
from ...core.labels import Labeler
from ...core.results import ResultsCollector, ResultWriter
from ...core.trans import ModelContainer, SeqClassifyModel


from ...nf.utils import compute_model_name, compute_model_path, write_model_params, get_labels, get_train_fields

logger = logging.getLogger('nf.train')


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


def _prep(arg, pt_method: str) -> Tuple[str, List[str]]:
    # compute a model name from the train params if not given
    arg.model_name = compute_model_name(arg, pt_method)
    # compute final model collection path based on model name
    fold = 't'
    if arg.k_fold > 0:
        fold = arg.k_fold
    result_path = os.path.join(compute_model_path(arg.result_dir, f'k{fold}.' + arg.model_name))
    # store all input parameters
    params = write_model_params(result_path, arg, pt_method)
    # determine which labels to use - all or just a subset
    labels = get_labels(arg)
    logger.info(
        'Starting training model [%s] with device [%s] for labels [%s] for params %s with result path [%s]...',
        arg.model_name, arg.device, labels, params, result_path
    )
    return result_path, labels


def _train(
        arg, labeler: Labeler, result_path: str, train_data: pd.DataFrame, eval_data: pd.DataFrame,
        eval_collector: Union[ResultsCollector, None] = None
) -> Tuple[ModelContainer, Dict[str, Any]]:
    text_fields = get_train_fields(arg)
    labels = labeler.source_labels()
    logger.info('Training for label(s): [%s] with device [%s]', labels, arg.device)
    mc = SeqClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        labeler=labeler,
        cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
        device=arg.device,
        best_metric=arg.metric
    )
    logger.debug('Constructing train data set [%s]...', len(train_data))
    train_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, labels, text_fields
    )
    logger.info('Constructed train data set [%s].', len(train_data))
    logger.debug('Constructing evaluation data set [%s]...', len(eval_data))
    eval_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, labels, text_fields
    )
    logger.info('Constructed evaluation data set [%s].', len(eval_data))

    training_args = _get_training_args(arg, result_path)
    # train the model
    if eval_collector is not None:
        results = mc.build(training_args, train_set, eval_set, eval_collector.collect)
    else:
        results = mc.build(training_args, train_set, eval_set)
    return mc, results


def _test(arg, mc: ModelContainer, result_path: str, test_data: pd.DataFrame,
          test_collector: Union[ResultsCollector, None]) -> Dict[str, Any]:
    text_fields = get_train_fields(arg)
    labels = mc.labeler().source_labels()
    # run tests
    logger.debug('Constructing test data set [%s]...', len(test_data))
    test_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, labels, text_fields
    )
    logger.info('Constructed test data set [%s].', len(test_data))
    results = mc.test(_get_training_args(arg, result_path), test_set, test_collector.collect)
    logger.info('Test set evaluation results: [%s].', results)

    return results


def _replace_binrel_models_tmp_dirs(arg, labels):
    for label in labels:
        path1 = os.path.join(
            compute_model_path(arg.result_dir, f'k{arg.k_fold}.' + arg.model_name), label + '-tmp'
        )
        path2 = os.path.join(
            compute_model_path(arg.result_dir, f'k{arg.k_fold}.' + arg.model_name), label
        )
        if os.path.exists(path2):
            shutil.rmtree(path2)
        shutil.move(path1, path2)


def _remove_binrel_models_tmp_dirs(arg, labels):
    for label in labels:
        path1 = os.path.join(
            compute_model_path(arg.result_dir, f'k{arg.k_fold}.' + arg.model_name), label + '-tmp'
        )
        if os.path.exists(path1):
            shutil.rmtree(path1)


def _remove_checkpoint_dir(result_path: str):
    for rd in os.listdir(result_path):
        checkpoint_path = os.path.join(result_path, rd)
        if not rd.startswith('checkpoint'):
            continue
        if not os.path.isdir(checkpoint_path):
            continue
        shutil.rmtree(checkpoint_path)


def _save_predictions(labels: List[str], data: pd.DataFrame, collector: Union[ResultsCollector, None]):
    if isinstance(labels, str) or len(labels) == 1:
        lbl = labels if isinstance(labels, str) else labels[0]
        data['p_' + lbl] = collector.y_pred
    else:
        for lx, lbl in enumerate(labels):
            data['p_' + lbl] = [item[lx] for item in collector.y_pred]


def _clear_predictions(labels: List[str], data: pd.DataFrame):
    p_labels = ['p_' + item for item in labels]
    data.drop(p_labels, axis=1, inplace=True)


def _fold_keep_model(arg, fold: int, best: Dict[str, Any], curr: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if best is None:
        return curr, True
    elif best['avg'][arg.metric]['f1'] < curr['avg'][arg.metric]['f1']:
        logger.info(
            'Training model [%s] fold [%s] metric [%s] value [%s] is better than previous [%s].',
            arg.model_name, fold, arg.metric,
            curr['avg'][arg.metric]['f1'],
            best['avg'][arg.metric]['f1']
        )
        return curr, True
    else:
        logger.info(
            'Training model [%s] fold [%s] metric [%s] value [%s] is worse than previous [%s].',
            arg.model_name, fold, arg.metric,
            curr['avg'][arg.metric]['f1'],
            best['avg'][arg.metric]['f1']
        )
        return best, True


def _write_results(arg, labels, result_path, result, df: pd.DataFrame, fold: Union[int, str] = 't',
                   drop_fields: Union[List[str], None] = None):
    # write predictions to model or model collection dir file kFOLD.predictions.csv
    if drop_fields is None:
        drop_fields = ['body', 'lead', 'embed_oai_ada2']

    result_writer = ResultWriter()
    result_writer.write_predictions(
        result_path, f'k{fold}.predictions', df, drop_fields
    )
    # remove predictions from the dataframe
    _clear_predictions(labels, df)
    # write metrics to model collection dir local file metrics.json and metrics.csv
    result_writer.write_metrics(
        result_path, 'metrics', f'k{fold}.' + arg.model_name, result
    )
    # write metrics to host global file
    result_writer.write_metrics(
        arg.result_dir, 'metrics_' + socket.gethostname(), f'k{fold}.' + arg.model_name, result
    )
