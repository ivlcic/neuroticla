from transformers import TrainingArguments

from neuroticla.core.dataset import SeqClassifyDataset
from neuroticla.core.labels import BinaryLabeler
from neuroticla.core.split import DataSplit
from neuroticla.core.trans import SeqClassifyModel, ModelContainer
from neuroticla.nf.utils import *

logger = logging.getLogger('nf.train.binrel')


def train_label(arg, label, train_data, eval_data, test_data):

    if arg.model_name is None:
        arg.model_name = arg.pretrained_model + '-' + arg.corpora + '-' + label

    result_path = compute_model_path(arg)

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

    logger.info('Training for label: %s', label)
    mc = SeqClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        BinaryLabeler(labels=[label]),
        os.path.join(arg.tmp_dir, arg.pretrained_model)
    )
    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, label, 'body'
    )
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, label, 'body'
    )
    logger.info("Constructed evaluation data set [%s].", len(eval_data))
    # train the model
    mc.build(training_args, train_set, eval_set)


def train(arg) -> 0:
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        train_label(arg, label, train_data, eval_data, test_data)
    return 0