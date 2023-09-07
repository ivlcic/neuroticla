import os.path

from transformers import TrainingArguments

from ...core.dataset import SeqClassifyDataset
from ...core.labels import BinaryLabeler
from ...core.results import ResultWriter
from ...core.split import DataSplit
from ...core.trans import SeqClassifyModel, ModelContainer
from ...nf.utils import *

logger = logging.getLogger('nf.train.binrel')


def train(arg) -> int:
    labels = get_labels('nf', arg)
    if not labels:
        return 1

    compute_m_name = arg.model_name is None

    train_data, eval_data, test_data = DataSplit.load(get_data_path_prefix(arg))
    for label in labels:
        compute_model_name(arg, [label], True)
        result_path = compute_model_path(arg)
        logger.info('Started training model [%s] for label [%s] to path [%s].',
                    arg.model_name, label, result_path)

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

        logger.info('Training for label: %s with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            ModelContainer.model_name_map[arg.pretrained_model],
            labeler=BinaryLabeler(labels=[label]),
            cache_model_dir=os.path.join(arg.tmp_dir, arg.pretrained_model),
            device=arg.device
        )
        logger.debug('Constructing train data set [%s]...', len(train_data))
        train_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), train_data, arg.max_seq_len, label, 'body'
        )
        logger.info('Constructed train data set [%s].', len(train_data))
        logger.debug('Constructing evaluation data set [%s]...', len(eval_data))
        eval_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), eval_data, arg.max_seq_len, label, 'body'
        )
        logger.info('Constructed evaluation data set [%s].', len(eval_data))
        # train the model
        mc.build(training_args, train_set, eval_set)

        # run tests
        logger.debug('Constructing test data set [%s]...', len(test_data))
        test_set = SeqClassifyDataset(
            mc.labeler(), mc.tokenizer(), test_data, arg.max_seq_len, label, 'body'
        )
        logger.info('Constructed test data set [%s].', len(test_data))
        results = mc.test(training_args, test_set)

        logger.info('Test set evaluation results:')
        logger.info('%s', results)

        # write results
        rw: ResultWriter = ResultWriter(arg.result_dir)
        rw.write(results, arg.model_name)
        logger.info('Writing [%s] to [%s].', arg.model_name, result_path)
        ModelContainer.remove_checkpoint_dir(result_path)

        # reset model name back to None to be recomputed
        if compute_m_name:
            arg.model_name = None

    return 0
