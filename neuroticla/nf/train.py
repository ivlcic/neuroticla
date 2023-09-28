import os.path
import socket
import shutil
import pandas as pd

from typing import Any, Tuple, Callable
from transformers import TrainingArguments
from sklearn.model_selection import KFold

from ..core.results import ResultsCollector
from ..core.dataset import SeqClassifyDataset
from ..core.labels import Labeler, BinaryLabeler, MultiLabeler
from ..core.results import ResultWriter
from ..core.split import DataSplit
from ..core.trans import SeqClassifyModel, ModelContainer
from ..core.eval import MultilabelMetrics
from ..nf.utils import *

logger = logging.getLogger('nf.train')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--max_seq_len', help='Max sub-word tokens length.', type=int, default=512
    )
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
    text_fields = ','.join(get_all_text_fields())
    parser.add_argument(
        '-f', '--train_fields', type=str, default='body', required=False,
        help='Text fields to use for training: ' + text_fields + ')',
    )
    parser.add_argument(
        '-p', '--pretrained_model', type=str, default=None, required=False,
        help='Pretrained model that should be used for fine tuning',
        choices=['mcbert', 'xlmrb', 'xlmrl']
    )
    parser.add_argument(
        '-m', '--metric', type=str, default='micro-1', required=False,
        help='Metric to select for best model selection (used only for model name construction)',
        choices=['micro-1', 'micro', 'macro-1', 'macro']
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

def _replace_binrel_models_tmp_dirs(arg, labels):
    for label in labels:
        path1 = os.path.join(compute_model_path(arg, 'binrel'), label + '-tmp')
        path2 = os.path.join(compute_model_path(arg, 'binrel'), label)
        if os.path.exists(path2):
            shutil.rmtree(path2)
        shutil.move(path1, path2)


def _remove_binrel_models_tmp_dirs(arg, labels):
    for label in labels:
        path1 = os.path.join(compute_model_path(arg, 'binrel'), label + '-tmp')
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
    results = mc.build(training_args, train_set, eval_set, eval_collector.collect)
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


def train_binrel(arg) -> int:
    logger.info('Starting binary relevance training ...')
    arg.model_name = compute_model_name(arg, 'binrel')
    result_path = os.path.join(compute_model_path(arg, 'binrel'))
    params = write_model_params(result_path, arg, 'binrel')
    labels = get_labels(arg)
    logger.info(
        'Starting training model [%s] with device [%s] for labels [%s] for params %s with result path [%s]...',
        arg.model_name, arg.device, labels, params, result_path
    )

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    if arg.k_fold == 0:
        collector = ResultsCollector()
        for label in labels:
            sub_result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
            logger.info('Started training model [%s] for label [%s] to path [%s].',
                        arg.model_name, label, sub_result_path)
            mc, _ = _train(
                arg, BinaryLabeler(labels=[label]), sub_result_path, train_data, eval_data, None
            )
            _test(
                arg, mc, sub_result_path, test_data, collector
            )
            _save_predictions(labels, test_data, collector)
            ModelContainer.remove_checkpoint_dir(sub_result_path)
            mc.destroy()

        metrics = MultilabelMetrics()
        result = metrics.compute(
            references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
        )

        result_writer = ResultWriter()
        result_writer.write_predictions(result_path, 'kt.predictions', test_data, ['body', 'lead'])
        _clear_predictions(labels, test_data)
        result_writer.write_metrics(result_path, 'kt.metrics', 'kt.' + arg.model_name, result, True)
        result_writer.write_metrics(arg.result_dir, 'results_' + socket.gethostname(), 'kt.' + arg.model_name, result)
    else:
        data = pd.concat([train_data, eval_data, test_data], ignore_index=True)
        kfold = KFold(n_splits=arg.k_fold)
        best_result = None
        for fold, (train_index, eval_index) in enumerate(kfold.split(data)):
            train_df = data.iloc[train_index]
            eval_df = data.iloc[eval_index]
            logger.info(
                'Training model [%s] fold [%s] with train size [%s] and validation size [%s]...',
                arg.model_name, fold, train_df.shape[0], eval_df.shape[0]
            )
            collector = ResultsCollector()
            for label in labels:
                sub_result_path = os.path.join(compute_model_path(arg, 'binrel'), label + '-tmp')
                logger.info('Started training model [%s] for label [%s] to path [%s].',
                            arg.model_name, label, sub_result_path)
                mc, _ = _train(
                    arg, BinaryLabeler(labels=[label]), sub_result_path, train_df, eval_df, collector
                )
                _save_predictions(labels, eval_df, collector)
                ModelContainer.remove_checkpoint_dir(sub_result_path)
                mc.destroy()

            metrics = MultilabelMetrics()
            result = metrics.compute(
                references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
            )
            if best_result is None:
                best_result = result
                _replace_binrel_models_tmp_dirs(arg, labels)
            elif best_result['avg'][arg.metric]['f1'] < result['avg'][arg.metric]['f1']:
                logger.info(
                    'Training model [%s] fold [%s] metric [%s] value [%s] is better than previous [%s].',
                    arg.model_name, fold, arg.metric,
                    result['avg'][arg.metric]['f1'],
                    best_result['avg'][arg.metric]['f1']
                )
                best_result = result
                _replace_binrel_models_tmp_dirs(arg, labels)
            else:
                logger.info(
                    'Training model [%s] fold [%s] metric [%s] value [%s] is worse than previous [%s].',
                    arg.model_name, fold, arg.metric,
                    result['avg'][arg.metric]['f1'],
                    best_result['avg'][arg.metric]['f1']
                )
                _remove_binrel_models_tmp_dirs(arg, labels)

            result_writer = ResultWriter()
            result_writer.write_predictions(
                result_path, f'k{fold}.predictions', eval_df, ['body', 'lead']
            )
            _clear_predictions(labels, eval_df)
            result_writer.write_metrics(
                result_path, 'metrics', f'k{fold}.' + arg.model_name, result, True
            )
            result_writer.write_metrics(
                arg.result_dir, 'metrics_' + socket.gethostname(), f'k{fold}.' + arg.model_name, result
            )

    logger.info('Finished binary relevance training.')
    return 0


def train_lpset(arg) -> int:
    logger.info("Starting label power-set training ...")

    arg.model_name = compute_model_name(arg, 'lpset')
    result_path = os.path.join(compute_model_path(arg, 'lpset'))
    params = write_model_params(result_path, arg, 'lpset')
    labels = get_labels(arg)
    logger.info(
        'Starting training model [%s] with device [%s] for labels [%s] for params %s with result path [%s]...',
        arg.model_name, arg.device, labels, params, result_path
    )

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    if arg.k_fold == 0:
        collector = ResultsCollector()
        mc, results = _train(
            arg, MultiLabeler(labels=labels), result_path, train_data, eval_data
        )
        result = _test(
            arg, mc, result_path, test_data, collector
        )
        _save_predictions(labels, test_data, collector)
        ModelContainer.remove_checkpoint_dir(result_path)

        result_writer = ResultWriter()
        result_writer.write_predictions(result_path, 'kt.predictions', test_data, ['body', 'lead'])
        _clear_predictions(labels, test_data)
        result_writer.write_metrics(result_path, 'kt.metrics', 'kt.' + arg.model_name, result, True)
        result_writer.write_metrics(arg.result_dir, 'metrics_' + socket.gethostname(), 'kt.' + arg.model_name, result)
    else:
        data = pd.concat([train_data, eval_data, test_data], ignore_index=True)
        kfold = KFold(n_splits=arg.k_fold)
        best_result = None
        for fold, (train_index, eval_index) in enumerate(kfold.split(data)):
            train_df = data.iloc[train_index]
            eval_df = data.iloc[eval_index]
            logger.info(
                'Training model [%s] fold [%s] with train size [%s] and validation size [%s]...',
                arg.model_name, fold, train_df.shape[0], eval_df.shape[0]
            )
            collector = ResultsCollector()
            mc, result = _train(
                arg, MultiLabeler(labels=labels), result_path, train_df, eval_df, collector
            )
            _save_predictions(labels, eval_df, collector)

            if best_result is None:
                best_result = result
                ModelContainer.remove_checkpoint_dir(result_path)
            elif best_result['avg'][arg.metric]['f1'] < result['avg'][arg.metric]['f1']:
                logger.info(
                    'Training model [%s] fold [%s] metric [%s] value [%s] is better than previous [%s].',
                    arg.model_name, fold, arg.metric,
                    result['avg'][arg.metric]['f1'],
                    best_result['avg'][arg.metric]['f1']
                )
                best_result = result
                ModelContainer.remove_checkpoint_dir(result_path)
            else:
                logger.info(
                    'Training model [%s] fold [%s] metric [%s] value [%s] is worse than previous [%s].',
                    arg.model_name, fold, arg.metric,
                    result['avg'][arg.metric]['f1'],
                    best_result['avg'][arg.metric]['f1']
                )
                _remove_checkpoint_dir(result_path)

            mc.destroy()

            result_writer = ResultWriter()
            result_writer.write_predictions(
                result_path, f'k{fold}.predictions', test_data, ['body', 'lead']
            )
            _clear_predictions(labels, eval_df)
            result_writer.write_metrics(
                result_path, 'metrics', f'k{fold}.' + arg.model_name, result, True
            )
            result_writer.write_metrics(
                arg.result_dir, 'metrics_' + socket.gethostname(), f'k{fold}.' + arg.model_name, result
            )

    logger.info('Finished label power-set training.')
    return 0
