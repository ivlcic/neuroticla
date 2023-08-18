import os
import json
import logging

from argparse import ArgumentParser
from typing import Dict, List, Union

from transformers import TrainingArguments

from neuroticla.core.args import CommonArguments
from neuroticla.core.dataset import TokenClassifyDataset
from neuroticla.core.json import NpEncoder
from neuroticla.core.labels import Labeler
from neuroticla.core.split import DataSplit
from neuroticla.core.trans import TokenClassifyModel, ModelContainer

logger = logging.getLogger('ner.train')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(nrcla_module, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(nrcla_module, parser, ('-t', '--tmp_dir'))
    CommonArguments.train(nrcla_module, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    parser.add_argument(
        '--no_misc', help='Remove MISC tag (replace i with "O").', action='store_true', default=False
    )
    parser.add_argument(
        '--pro', help='Enable Product (PRO) tag.', action='store_true', default=False
    )
    parser.add_argument(
        '--evt', help='Enable Event (EVT) tag.', action='store_true', default=False
    )
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    parser.add_argument(
        'langs', help='Languages to train', nargs='+',
        choices=['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'bg', 'pl', 'ru', 'sk', 'uk']
    )


def pretrained_model_path(args, train: bool = False) -> Union[str, List[str]]:
    if train:
        pt_model_dir = os.path.join(args.tmp_dir, args.pretrained_model)
        if not os.path.exists(pt_model_dir):
            os.makedirs(pt_model_dir)
        return pt_model_dir
    else:
        if isinstance(args.pretrained_model, list):
            ret = []
            for m in args.pretrained_model:
                ret.append(os.path.join(args.models_dir, m))
            return ret
        elif isinstance(args.pretrained_model, str):
            return os.path.join(args.models_dir, args.pretrained_model)
        else:
            raise ValueError('Unsupported args.pretrained_model type!')


def replace_ner_tags(args) -> Dict[str, str]:
    del_misc = {}
    if hasattr(args, 'no_misc') and args.no_misc:
        del_misc['B-MISC'] = 'O'
        del_misc['I-MISC'] = 'O'
    if not hasattr(args, 'pro') or not args.pro:
        del_misc['B-PRO'] = 'O'
        del_misc['I-PRO'] = 'O'
    if not hasattr(args, 'evt') or not args.evt:
        del_misc['B-EVT'] = 'O'
        del_misc['I-EVT'] = 'O'
    return del_misc


def main(arg) -> int:
    corpora_prefix = '_'.join(arg.langs)
    if arg.model_name is None:
        model_name = arg.pretrained_model + '-' + corpora_prefix
        if arg.no_misc:
            model_name += '-nomisc'
        arg.model_name = model_name

    result_path = os.path.join(arg.data_out_dir, arg.model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    training_args = TrainingArguments(
        output_dir=result_path,
        num_train_epochs=arg.epochs,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        evaluation_strategy="epoch",
        disable_tqdm=True,
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
        pretrained_model_path(arg, train=True)
    )

    label_field = 'ner'
    text_field = 'sentence'
    data_path = os.path.join(arg.data_in_dir, corpora_prefix)

    # load the data and tokenize it

    train_data, eval_data, test_data = DataSplit.load([data_path])
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
    combined_results = {}
    if os.path.exists(os.path.join(arg.models_dir, 'results_all.json')):
        with open(os.path.join(arg.data_out_dir, 'results_all.json')) as json_file:
            combined_results = json.load(json_file)

    combined_results[arg.model_name] = results
    with open(os.path.join(arg.models_dir, 'results_all.json'), 'wt', encoding='utf-8') as fp:
        json.dump(combined_results, fp, cls=NpEncoder)
    with open(os.path.join(arg.models_dir, arg.model_name + ".json"), 'wt') as fp:
        json.dump(results, fp, cls=NpEncoder)
    return 0
