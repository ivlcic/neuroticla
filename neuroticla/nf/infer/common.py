import logging
from typing import Union, List

import pandas as pd
from transformers import TrainingArguments

from ...core.dataset import SeqClassifyDataset
from ...core.trans import SeqClassifyModel

logger = logging.getLogger('nf.infer')


def run_inference(arg, mc: SeqClassifyModel, infer_args: TrainingArguments, data: pd.DataFrame,
                  label: Union[str, List[str]] = 'label', text_field: Union[str, List[str]] = 'body'):
    # run tests
    logger.debug('Constructing inference data set [%s]...', len(data))
    data_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), data, arg.max_seq_len, label, text_field
    )
    logger.info('Constructed inference data set [%s].', len(data))
    results = mc.infer_set(infer_args, data_set)

    logger.info('Data set evaluation results [%s]', len(results))
    return results