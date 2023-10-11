import logging

from transformers import TrainingArguments

from ..core.dataset import TokenClassifyDataset
from ..core.labels import Labeler
from ..core.split import DataSplit
from ..core.trans import TokenClassifyModel
from ..ner.utils import *

logger = logging.getLogger('ner.test')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.test(parser, 32)
    add_common_test_train_args(module_name, parser)
    parser.add_argument(
        'model_name', help='Model name or path to test.', type=str, default=None
    )
    parser.add_argument(
        'langs', help='Languages to test against.', nargs='+',
        choices=get_all_languages()
    )


def main(arg) -> int:
    result_path = compute_model_path(arg)

    mc: TokenClassifyModel = TokenClassifyModel(
        result_path,
        labeler=Labeler(
            os.path.join(CommonArguments.data_path('ner', 'processed'), 'tags.csv'),
            replace_labels=replace_ner_tags(arg)
        ),
        device=arg.device
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
    test_set = TokenClassifyDataset(
        mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, 'ner', 'sentence'
    )
    results = mc.test(testing_args, test_set)
    logger.info("Test set evaluation results:")
    logger.info("%s", results)
    return 0
