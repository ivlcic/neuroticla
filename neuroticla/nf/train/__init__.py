import os
import logging
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import KFold

from ...core.args import CommonArguments
from ...core.eval import MultilabelMetrics
from ...core.labels import BinaryLabeler, MultiLabeler
from ...core.results import ResultsCollector
from ...core.split import DataSplit
from ...core.trans import ModelContainer
from .common import _prep, _train, _test, _save_predictions, _write_results, _fold_keep_model, \
    _replace_binrel_models_tmp_dirs, _remove_binrel_models_tmp_dirs, _remove_checkpoint_dir
from ..utils import get_all_labels, get_all_text_fields, get_data_path_prefix, compute_model_path

logger = logging.getLogger('nf.train')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser, 24, 512)
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


def train_binrel(arg) -> int:
    logger.info('Starting binary relevance training ...')
    result_path, labels = _prep(arg, 'binrel')

    # do a classical split
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    if arg.k_fold == 0:
        # for each label we'll train a model, and we'll store the predictions from the test-set
        collector = ResultsCollector()
        for label in labels:
            sub_result_path = os.path.join(
                result_path, label
            )
            logger.info('Started training model [%s] for label [%s] to path [%s].',
                        arg.model_name, label, sub_result_path)
            # train the model for a single label
            mc, _ = _train(
                arg, BinaryLabeler(labels=[label]), sub_result_path, train_data, eval_data
            )
            # test the model and compute predictions
            _test(
                arg, mc, sub_result_path, test_data, collector
            )
            # write predictions to the test-set dataframe
            _save_predictions([label], test_data, collector)
            # move files from the check-point dir to the LABEL model dir
            ModelContainer.remove_checkpoint_dir(sub_result_path)
            # free CUDA memory
            mc.destroy()

        # compute metrics from all single-label models
        metrics = MultilabelMetrics()
        result = metrics.compute(
            references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
        )
        _write_results(arg, labels, result_path, result, test_data)
    else:
        # we combine back the data and compute k-folds
        data = pd.concat([train_data, eval_data, test_data], ignore_index=True)
        kfold = KFold(n_splits=arg.k_fold)
        best_result = None
        # for each fold we'll train a model for each label and compute evaluation
        for fold, (train_index, eval_index) in enumerate(kfold.split(data)):
            train_df = data.iloc[train_index].copy()  # train dataframe
            eval_df = data.iloc[eval_index].copy()  # validation dataframe
            logger.info(
                'Training model [%s] fold [%s] with train size [%s] and validation size [%s]...',
                arg.model_name, fold, train_df.shape[0], eval_df.shape[0]
            )
            # for each label we'll train a model, and we'll store the predictions from the eval-set
            collector = ResultsCollector()
            for label in labels:
                sub_result_path = os.path.join(
                    compute_model_path(arg.result_dir, f'k{arg.k_fold}.' + arg.model_name), label + '-tmp'
                )
                logger.info('Started training model [%s] for label [%s] to path [%s].',
                            arg.model_name, label, sub_result_path)
                # train the model for a single label
                mc, _ = _train(
                    arg, BinaryLabeler(labels=[label]), sub_result_path, train_df, eval_df, collector
                )
                # write predictions to the eval-set dataframe
                _save_predictions([label], eval_df, collector)
                # move files from the check-point dir to the LABEL-tmp model dir
                ModelContainer.remove_checkpoint_dir(sub_result_path)
                # free CUDA memory
                mc.destroy()

            metrics = MultilabelMetrics()
            result = metrics.compute(
                references=collector.get_all_true(), predictions=collector.get_all_pred(), labels=labels
            )
            # do we keep the trained model?
            best_result, keep = _fold_keep_model(arg, fold, best_result, result)
            if keep:
                # promote LABEL-tmp model dirs to LABEL dirs
                _replace_binrel_models_tmp_dirs(arg, labels)
            else:
                # remove LABEL-tmp model dirs since we're doing worse
                _remove_binrel_models_tmp_dirs(arg, labels)

            _write_results(arg, labels, result_path, result, eval_df, fold)

    logger.info('Finished binary relevance training.')
    return 0


def train_lpset(arg) -> int:
    logger.info("Starting label power-set training ...")
    result_path, labels = _prep(arg, 'lpset')
    # do a classical split
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    if arg.k_fold == 0:
        mc, results = _train(
            arg, MultiLabeler(labels=labels), result_path, train_data, eval_data
        )
        # we'll store the predictions from the test-set
        collector = ResultsCollector()
        # test the model and compute predictions
        result = _test(
            arg, mc, result_path, test_data, collector
        )
        # write predictions to the test-set dataframe
        _save_predictions(labels, test_data, collector)
        # move files from the check-point dir to the model dir
        ModelContainer.remove_checkpoint_dir(result_path)

        _write_results(arg, labels, result_path, result, test_data)
    else:
        # we combine back the data and compute k-folds
        data = pd.concat([train_data, eval_data, test_data], ignore_index=True)
        kfold = KFold(n_splits=arg.k_fold)
        best_result = None
        # for each fold we'll train a model and compute evaluation
        for fold, (train_index, eval_index) in enumerate(kfold.split(data)):
            train_df = data.iloc[train_index].copy()
            eval_df = data.iloc[eval_index].copy()
            logger.info(
                'Training model [%s] fold [%s] with train size [%s] and validation size [%s]...',
                arg.model_name, fold, train_df.shape[0], eval_df.shape[0]
            )
            # train the model and store eval-set predictions
            collector = ResultsCollector()
            mc, result = _train(
                arg, MultiLabeler(labels=labels), result_path, train_df, eval_df, collector
            )
            # write predictions to the eval-set dataframe
            _save_predictions(labels, eval_df, collector)
            # do we keep the trained model?
            best_result, keep = _fold_keep_model(arg, fold, best_result, result)
            if keep:
                # promote model checkpoint dir to model dir (keep the model)
                ModelContainer.remove_checkpoint_dir(result_path)
            else:
                # remove model checkpoint dir (delete the model)
                _remove_checkpoint_dir(result_path)
            # free CUDA memory
            mc.destroy()

            _write_results(arg, labels, result_path, result, eval_df, fold)

    logger.info('Finished label power-set training.')
    return 0
