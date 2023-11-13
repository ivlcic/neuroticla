import json
import logging
import os
from argparse import ArgumentParser
from typing import List, Union, Dict

from ..core.args import CommonArguments

logger = logging.getLogger('nf.utils')


def get_all_labels(corpora: Union[str, None] = None) -> List[str]:
    labels_file = os.path.join(
        CommonArguments.data_path('nf', 'processed'), 'tags.csv'
    )
    if corpora is None:
        corpora = ''
    corpora_labels_file = os.path.join(
        CommonArguments.data_path('nf', 'processed'), corpora + '.tags.csv'
    )
    if os.path.exists(corpora_labels_file):
        labels = open(corpora_labels_file, 'r').read().split('\n')
    else:
        labels = open(labels_file, 'r').read().split('\n')
    return labels


def get_labels(arg) -> List[str]:
    labels = get_all_labels(arg.corpora)
    if arg.subset is not None:
        subset = arg.subset.split(',')
        if all(item in labels for item in subset):
            labels = subset
        else:
            raise ValueError(f'Invalid labels subset: [{arg.subset}]')
    return labels


def get_all_text_fields() -> List[str]:
    return ['title', 'body']


def get_all_embeddings_fields() -> List[str]:
    return ['embed_oai_ada2']


def get_text_fields(input_fields: str) -> List[str]:
    text_fields = get_all_text_fields()
    if input_fields is not None:
        fields = input_fields.split(',')
        if all(item in text_fields for item in fields):
            text_fields = fields
        else:
            raise ValueError(f'Invalid text fields: [{input_fields}]')
    return text_fields


def get_train_fields(arg) -> List[str]:
    return get_text_fields(arg.train_fields)


def get_test_fields(arg) -> List[str]:
    if arg.test_fields is None:
        return get_text_fields(arg.train_fields)
    return get_text_fields(arg.test_fields)


def get_data_path_prefix(arg) -> List[str]:
    data_paths = [os.path.join(arg.data_in_dir, arg.corpora)]
    return data_paths


def get_labels_str(labels: List[str]) -> str:
    l_str = ''
    if labels is not None:
        l_str = '.l-' + '_'.join(labels)
    return l_str


def compute_model_name(arg, pt_method: str, nn: bool = True) -> str:
    if arg.model_name is not None:
        return arg.model_name
    l_str = get_labels_str(get_labels(arg))
    f_str = ''
    if nn:
        train_fields = get_train_fields(arg)
    else:
        train_fields = arg.train_fields

    if train_fields is not None:
        if isinstance(train_fields, str):
            f_str = '.f-' + train_fields
        else:
            f_str = '.f-' + '_'.join([text_field[0] for text_field in train_fields if text_field])

    if nn:
        m = (f'{pt_method}.{arg.pretrained_model}.e{arg.epochs}.b{arg.batch}'
             f'.l{arg.learn_rate}.m-{arg.metric}.{arg.corpora}{f_str}{l_str}')
    else:
        m = f'{pt_method}.{arg.algorithm}.m-{arg.metric}.{arg.corpora}{f_str}{l_str}'
    return m


def write_model_params(out_dir: str, arg, pt_method: str, nn: bool = True) -> Dict[str, str]:
    params = {}
    params['pt_method'] = pt_method
    params['name'] = compute_model_name(arg, pt_method, nn)
    params['labels'] = ','.join(get_labels(arg))
    params['metric'] = arg.metric
    params['corpora'] = arg.corpora
    if nn:
        params['train_fields'] = ','.join(get_train_fields(arg))
        params['pretrained_model'] = arg.pretrained_model
        params['epochs'] = arg.epochs
        params['batch'] = arg.batch
        params['learn_rate'] = arg.learn_rate
        params['max_seq_len'] = arg.max_seq_len
    else:
        params['train_fields'] = arg.train_fields
        params['algorithm'] = arg.algorithm

    with open(os.path.join(out_dir, 'train_params.json'), 'wt') as fp:
        json.dump(params, fp, indent=2)
    return params


def read_model_params(out_dir: str, arg) -> Dict[str, str]:
    with open(os.path.join(out_dir, 'train_params.json'), 'r') as json_file:
        params = json.load(json_file)

    arg.labels = params['labels']
    arg.train_fields = params['train_fields']
    arg.pretrained_model = params['pretrained_model']
    arg.epochs = params['epochs']
    arg.batch = params['batch']
    arg.learn_rate = params['learn_rate']
    arg.metric = params['metric']
    arg.corpora = params['corpora']
    arg.max_seq_len = params['max_seq_len']
    return params


def compute_model_path(result_path: str, model_dir: str) -> str:
    if not os.path.exists(model_dir):
        result_path = os.path.join(result_path, model_dir)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        result_path = model_dir
    return result_path


def add_common_test_train_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--max_seq_len', help='Max sub-word tokens length.', type=int, default=512
    )
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
