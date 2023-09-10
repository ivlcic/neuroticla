import logging
import pandas as pd

from typing import Union, List, Callable

from transformers import TrainingArguments

from ..utils import compute_model_name
from ...core.dataset import SeqClassifyDataset
from ...core.results import ResultWriter
from ...core.trans import SeqClassifyModel

logger = logging.getLogger('nf.test')


def run_test(arg, mc: SeqClassifyModel, test_args: TrainingArguments, test_data: pd.DataFrame,
             label: Union[str, List[str]] = 'label',
             text_field: Union[str, List[str]] = 'body',
             callback: Callable = None):
    # run tests
    logger.debug('Constructing test data set [%s]...', len(test_data))
    test_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, label, text_field
    )
    logger.info('Constructed test data set [%s].', len(test_data))
    results = mc.test(test_args, test_set, callback)

    logger.info('Test set evaluation results:')
    logger.info('%s', results)
    return results


def write_test_results(arg, results, label: List[str]):
    # write results
    result_name = compute_model_name(arg, label, True)
    rw: ResultWriter = ResultWriter(arg.result_dir)
    rw.write(results, result_name)
    logger.info('Written test results [%s] to [%s].', arg.model_name, arg.result_dir)
