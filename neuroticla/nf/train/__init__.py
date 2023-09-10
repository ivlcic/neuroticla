from transformers import TrainingArguments

from ..test import run_test, write_test_results
from ...core.dataset import SeqClassifyDataset
from ...core.labels import BinaryLabeler, MultiLabeler
from ...core.split import DataSplit
from ...core.trans import SeqClassifyModel, ModelContainer
from ...nf.utils import *

logger = logging.getLogger('nf.train')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(module_name, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    labels = ','.join(get_all_labels(module_name))
    parser.add_argument(
        '-u', '--subset', type=str, default=None,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])

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
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    text_field = 'body'
    compute_m_name = arg.model_name is None

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        arg.model_name = compute_model_name(arg, None, True)  # model name w/o label
        result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started training model [%s] for label [%s] to path [%s].',
                    arg.model_name, label, result_path)

        logger.info('Training for label: %s with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            ModelContainer.model_name_map[arg.pretrained_model],
            labeler=BinaryLabeler(labels=[label]),
            cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
            device=arg.device
        )
        logger.debug('Constructing train data set [%s]...', len(train_data))
        train_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, label, text_field
        )
        logger.info('Constructed train data set [%s].', len(train_data))
        logger.debug('Constructing evaluation data set [%s]...', len(eval_data))
        eval_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, label, text_field
        )
        logger.info('Constructed evaluation data set [%s].', len(eval_data))

        training_args = _get_training_args(arg, result_path)
        # train the model
        mc.build(training_args, train_set, eval_set)

        results = run_test(arg, mc, training_args, test_data, label)
        write_test_results(arg, results, [label])
        logger.info('Writing [%s] to [%s].', arg.model_name, result_path)
        ModelContainer.remove_checkpoint_dir(result_path)

        # reset model name back to None (if that was at the beginning) to be recomputed
        if compute_m_name:
            arg.model_name = None

    return 0


def train_lpset(arg) -> int:
    logger.info("Starting label power-set ...")
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))

    compute_model_name(arg, labels)
    result_path = compute_model_path(arg, 'lpset')

    logger.info('Training for labels: %s with device [%s]', labels, arg.device)
    mc = SeqClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        labeler=MultiLabeler(labels=labels),
        cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
        device=arg.device
    )

    text_field = 'body'

    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, labels, text_field
    )
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, labels, text_field
    )
    logger.info("Constructed evaluation data set [%s].", len(eval_data))

    training_args = _get_training_args(arg, result_path)
    # train the model
    mc.build(training_args, train_set, eval_set)

    results = run_test(arg, mc, training_args, test_data, labels)
    write_test_results(arg, results, labels)

    ModelContainer.remove_checkpoint_dir(result_path)
    return 0
