import logging

from transformers import TrainingArguments

from ..core.dataset import TokenClassifyDataset
from ..core.labels import Labeler
from ..core.results import ResultWriter
from ..core.split import DataSplit
from ..core.trans import TokenClassifyModel, ModelContainer
from ..ner.utils import *

logger = logging.getLogger('ner.train')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser, 32, 256)
    add_common_test_train_args(module_name, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    parser.add_argument(
        'langs', help='Languages to use.', nargs='+',
        choices=get_all_languages()
    )


def main(arg) -> int:
    corpora_prefix = '_'.join(arg.langs)
    if arg.model_name is None:
        model_name = arg.pretrained_model + '-' + corpora_prefix
        if arg.no_misc:
            model_name += '-nomisc'
        arg.model_name = model_name

    result_path = compute_model_path(arg)

    mc = TokenClassifyModel(
        ModelContainer.model_name_map[arg.pretrained_model],
        labeler=Labeler(
            os.path.join(CommonArguments.data_path('ner', 'processed'), 'tags.csv'),
            replace_labels=replace_ner_tags(arg)
        ),
        cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
        device=arg.device
    )

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

    label_field = 'ner'
    text_field = 'sentence'

    train_data, eval_data, test_data = DataSplit.load(get_data_paths_prefixes(arg))
    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = TokenClassifyDataset(
        mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, label_field, text_field
    )
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = TokenClassifyDataset(
        mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, label_field, text_field
    )
    logger.info("Constructed evaluation data set [%s].", len(eval_data))
    # train the model
    mc.build(training_args, train_set, eval_set)

    # run tests
    logger.debug("Constructing test data set [%s]...", len(test_data))
    test_set = TokenClassifyDataset(
        mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, label_field, text_field
    )
    logger.info("Constructed test data set [%s].", len(test_data))
    results = mc.test(training_args, test_set)

    logger.info("Test set evaluation results:")
    logger.info("%s", results)

    # write results
    rw: ResultWriter = ResultWriter()
    rw.write_metrics(result_path, arg.model_name, arg.model_name, results, True)

    ModelContainer.remove_checkpoint_dir(result_path)
    return 0
