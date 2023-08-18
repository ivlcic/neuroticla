import json
import logging

from transformers import TrainingArguments

from neuroticla.core.dataset import TokenClassifyDataset
from neuroticla.core.json import NpEncoder
from neuroticla.core.labels import Labeler
from neuroticla.core.split import DataSplit
from neuroticla.core.trans import TokenClassifyModel, ModelContainer
from neuroticla.ner.utils import *

logger = logging.getLogger('ner.train')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.train(nrcla_module, parser)
    add_common_test_train_args(nrcla_module, parser)


def main(arg) -> int:
    compute_model_name(arg)

    result_path = os.path.join(arg.result_dir, arg.model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

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

    mc = TokenClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        Labeler(
            os.path.join(CommonArguments.data_path('ner', 'processed'), 'tags.csv'),
            replace_labels=replace_ner_tags(arg)
        ),
        os.path.join(arg.tmp_dir, arg.pretrained_model)
    )

    label_field = 'ner'
    text_field = 'sentence'
    # load the data and tokenize it

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = TokenClassifyDataset(mc.labeler(), mc.tokenizer(), train_data,
                                     arg.max_seq_len, label_field, text_field)
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = TokenClassifyDataset(mc.labeler(), mc.tokenizer(), eval_data,
                                    arg.max_seq_len, label_field, text_field)
    logger.info("Constructed evaluation data set [%s].", len(eval_data))
    # train the model
    mc.build(training_args, train_set, eval_set)

    # run tests
    logger.debug("Constructing test data set [%s]...", len(test_data))
    test_set = TokenClassifyDataset(mc.labeler(), mc.tokenizer(), test_data,
                                    arg.max_seq_len, label_field, text_field)
    logger.info("Constructed test data set [%s].", len(test_data))
    results = mc.test(training_args, test_set)

    logger.info("Test set evaluation results:")
    logger.info("%s", results)

    # write results
    combined_results = {}
    if os.path.exists(os.path.join(arg.result_dir, 'results_all.json')):
        with open(os.path.join(arg.result_dir, 'results_all.json')) as json_file:
            combined_results = json.load(json_file)

    combined_results[arg.model_name] = results
    with open(os.path.join(arg.result_dir, 'results_all.json'), 'wt', encoding='utf-8') as fp:
        json.dump(combined_results, fp, cls=NpEncoder)
    with open(os.path.join(arg.result_dir, arg.model_name + ".json"), 'wt') as fp:
        json.dump(results, fp, cls=NpEncoder)
    return 0
