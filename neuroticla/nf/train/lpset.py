import logging
import os

from transformers import TrainingArguments

from ...core.dataset import SeqClassifyDataset
from ...core.labels import MultiLabeler
from ...core.split import DataSplit
from ...core.results import ResultWriter
from ...core.trans import SeqClassifyModel, ModelContainer
from ...nf.utils import compute_model_name, compute_model_path, get_data_path_prefix, get_labels

logger = logging.getLogger('nf.train.lpset')


def train(arg) -> int:
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    # load the data and tokenize it
    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))

    arg.model_name = compute_model_name(arg, labels)
    result_path = compute_model_path(arg)

    training_args = TrainingArguments(
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

    logger.info('Training for labels: %s', labels)
    mc = SeqClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        MultiLabeler(labels=labels),
        os.path.join(arg.tmp_dir, arg.pretrained_model)
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

    # train the model
    mc.build(training_args, train_set, eval_set)

    # run tests
    logger.debug("Constructing test data set [%s]...", len(test_data))
    test_set = SeqClassifyDataset(
        mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, labels, text_field
    )
    logger.info("Constructed test data set [%s].", len(test_data))
    results = mc.test(training_args, test_set)

    logger.info("Test set evaluation results:")
    logger.info("%s", results)
    # write results
    rw: ResultWriter = ResultWriter(arg.result_dir)
    rw.write(results, arg.model_name)

    ModelContainer.remove_checkpoint_dir(result_path)
    return 0
