import logging

from transformers import TrainingArguments

from neuroticla.core.dataset import TokenClassifyDataset
from neuroticla.core.labels import Labeler
from neuroticla.core.split import DataSplit
from neuroticla.core.trans import TokenClassifyModel
from neuroticla.ner.utils import *

logger = logging.getLogger('ner.test')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.test(nrcla_module, parser)
    add_common_test_train_args(nrcla_module, parser)


def main(arg) -> int:
    compute_model_name(arg)

    result_path = os.path.join(arg.result_dir, arg.model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    mc: TokenClassifyModel = TokenClassifyModel(
        result_path,
        Labeler(
            os.path.join(CommonArguments.data_path('ner', 'processed'), 'tags.csv'),
            replace_labels=replace_ner_tags(arg)
        )
    )
    mc.eval()

    testing_args = TrainingArguments(
        output_dir=arg.result_dir,
        disable_tqdm=not arg.tqdm,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
    )
    logger.info("Starting test set evaluation...")

    _, _, test_data = DataSplit.load(get_data_paths_prefixes(arg))
    test_set = TokenClassifyDataset(mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, 'ner', 'sentence')
    results = mc.test(testing_args, test_set)
    logger.info("Test set evaluation results:")
    logger.info("%s", results)
    return 0
