import json
import logging

from transformers import TrainingArguments

from neuroticla.core.dataset import SeqClassifyDataset
from neuroticla.core.json import NpEncoder
from neuroticla.core.labels import Labeler, BinaryLabeler
from neuroticla.core.split import DataSplit
from neuroticla.core.trans import TokenClassifyModel, ModelContainer
from neuroticla.nf.utils import *

logger = logging.getLogger('nf.train')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(nrcla_module, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    labels = ','.join(get_all_labels(nrcla_module))
    parser.add_argument(
        '-u', '--subset', type=str, default=None,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])

    parser.add_argument('corpora', type=str, default=None,
                        help='Corpora prefix or path prefix to use for training')


def train_binrel(arg) -> int:
    logger.info('Starting binary relevance training ...')
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    training_args = TrainingArguments(
        output_dir=result_path,
        num_train_epochs=arg.epochs,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        evaluation_strategy="epoch",
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

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        logger.info('Training for label: %s', label)
        mc = SeqClassifyModel(
            ModelContainer.model_name_map[arg.pretrained_model],
            BinaryLabeler(labels=[label]),
            os.path.join(arg.tmp_dir, arg.pretrained_model)
        )
        logger.debug("Constructing train data set [%s]...", len(train_data))
        train_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, label_field, text_field
        )
        logger.info("Constructed train data set [%s].", len(train_data))
        logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
        eval_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, label_field, text_field
        )
        logger.info("Constructed evaluation data set [%s].", len(eval_data))
        # train the model
        mc.build(training_args, train_set, eval_set)
    return 0


def train_lpset(arg) -> int:
    logger.info("Starting label powerset ...")
    return 0
