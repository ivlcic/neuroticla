import logging
import os
import zipfile

from .bsnlp import default_conf as bsnlp_default_conf
from .bsnlp import to_csv as bsnlp_to_csv
from .conll import to_csv as conll_to_csv
from .wikiann import default_conf as wikiann_default_conf
from .wikiann import clean as wikiann_clean
from .cnec import clean as cnec_clean

from argparse import ArgumentParser
from typing import Dict, List, Any
from ...core.args import CommonArguments
from ...ner.utils import get_all_languages

logger = logging.getLogger('ner.prep')


def check_param(conf: Dict, p_name: str) -> Any:
    p = conf.get(p_name)
    if not p:
        logger.warning('Missing [%s] param in [%s] config', p_name, conf)
        exit(1)
    return p


def check_dir_param(conf: Dict, param_name: str, parent_path: str) -> str:
    file_name = check_param(conf, param_name)
    fpath = os.path.join(parent_path, file_name)
    if not os.path.exists(fpath):
        logger.warning('Missing [%s] filename in dir [%s]', file_name, parent_path)
        exit(1)
    return fpath


def merge_files_in_dir(args, conf: Dict[str, Any], files: List[str], suffix: str) -> str:
    proc_fname = check_dir_param(conf, 'proc_file', args.tmp_dir)
    data = ''
    for file_name in files:
        if data != '':
            data += "\n"
        with open(os.path.join(proc_fname, file_name), 'rt', encoding='utf-8') as fp:
            data += fp.read()
    proc_fname = os.path.join(proc_fname, args.lang + suffix)
    with open(proc_fname, 'wt', encoding='utf-8') as fp:
        fp.write(data)
    return proc_fname


def prep_data(args, confs: List[Dict]) -> None:
    for conf in confs:
        conf_type = conf.get('type')
        if conf_type == 'wikiann':
            ner_conll_idx = check_param(conf, 'ner_conll_idx')
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            zip_fname = check_dir_param(conf, 'zip', args.data_in_dir)
            zip_dir = os.path.join(args.tmp_dir, target_base_name)
            if not os.path.exists(zip_dir):
                os.mkdir(zip_dir)
            zipfile.ZipFile(zip_fname).extractall(zip_dir)
            args.process_file_name = merge_files_in_dir(
                args, conf, ['train', 'dev', 'test', 'extra'], '-wann.conll'
            )
            args.target_base_name = target_base_name
            wikiann_clean(args)
            conll_to_csv(args, ner_conll_idx, map_filter)
        if conf_type == 'cnec':
            ner_conll_idx = check_param(conf, 'ner_conll_idx')
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            zip_fname = check_dir_param(conf, 'zip', args.data_in_dir)
            zipfile.ZipFile(zip_fname).extractall(args.tmp_dir)
            args.process_file_name = merge_files_in_dir(
                args, conf, ['dtest.conll', 'train.conll', 'etest.conll'], '-cnec.conll'
            )
            args.target_base_name = target_base_name
            cnec_clean(args)
            conll_to_csv(args, ner_conll_idx, map_filter)
        if conf_type == 'conll':
            zip_fname = check_dir_param(conf, 'zip', args.data_in_dir)
            zipfile.ZipFile(zip_fname).extractall(args.tmp_dir)
            proc_fname = check_dir_param(conf, 'proc_file', args.tmp_dir)
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            logger.debug('Converting conll data [%s -> %s]...', proc_fname, target_base_name)
            ner_conll_idx = check_param(conf, 'ner_conll_idx')
            args.process_file_name = proc_fname
            args.target_base_name = target_base_name
            conll_to_csv(args, ner_conll_idx, map_filter)
            logger.info('Converted data [%s -> %s]', proc_fname, target_base_name)
        if conf_type == 'bsnlp':
            zip_fname = check_dir_param(conf, 'zip', args.data_in_dir)
            zipfile.ZipFile(zip_fname).extractall(args.tmp_dir)
            proc_fname = check_dir_param(conf, 'proc_file', args.tmp_dir)
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            logger.debug('Converting BSNLP data [%s -> %s]...', proc_fname, target_base_name)
            args.process_file_name = proc_fname
            args.target_base_name = target_base_name
            bsnlp_to_csv(args, map_filter)
            logger.info('Converted data [%s -> %s]', proc_fname, target_base_name)


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.processed_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '-a', '--append', help="Append to an existing CoNLL and CSV files", action='store_true'
    )
    parser.add_argument(
        '-p', '--password', type=str, default='showeffort',
        help="Zip file password",
    )
    parser.add_argument(
        'lang', help='language of the text', choices=get_all_languages()
    )


def main(args) -> int:
    if args.lang == 'sl':
        confs = [
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'ssj500k-syn.ud.conllu'),
                'result_name': args.lang + '_500k',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 256,
                    # 'stop_at': 9483,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'senticoref.ud.conllu'),
                'result_name': args.lang + '_scr',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 256,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'elexiswsd.ud.conllu'),
                'result_name': args.lang + '_ewsd',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 256,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
        # nf.data.multi_split_data(args, confs)
    if args.lang == 'hr':
        confs = [
            {
                'type': 'conll',
                'zip': 'hr500k-1.0.zip',
                'proc_file': os.path.join('hr500k.conll', 'hr500k.conll'),
                'result_name': args.lang + '_500k',
                'ner_conll_idx': 10,
                'map_filter': {
                    'max_seq_len': 256,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'sr':
        confs = [
            {
                'type': 'conll',
                'zip': 'setimes-sr.conll.zip',
                'proc_file': os.path.join('setimes-sr.conll', 'set.sr.conll'),
                'result_name': args.lang + '_set',
                'ner_conll_idx': 10,
                'map_filter': {
                    'max_seq_len': 256,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            }
        ]
        prep_data(args, confs)
    if args.lang == 'cs':
        confs = [
            {
                'type': 'cnec',
                'zip': 'CNEC_2.0_konkol.zip',
                'proc_file': 'CNEC_2.0_konkol',
                'result_name': args.lang + '_cnec',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 256,
                    'lang': args.lang,
                    'B-G': 'B-LOC', 'I-G': 'I-LOC',
                    'B-I': 'B-ORG', 'I-I': 'I-ORG',
                    'B-M': 'B-ORG', 'I-M': 'I-ORG',
                    'B-P': 'B-PER', 'I-P': 'I-PER',
                    'B-O': 'B-MISC', 'I-O': 'I-MISC',
                    'B-T': 'O', 'I-T': 'O',
                    'B-A': 'O', 'I-A': 'O'
                }
            },
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'bg':
        confs = [
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'ru':
        confs = [
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'pl':
        confs = [
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'sk':
        confs = [
            bsnlp_default_conf(args),
            wikiann_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'uk':
        confs = [
            bsnlp_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'bs':
        confs = [
            wikiann_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'mk':
        confs = [
            wikiann_default_conf(args)
        ]
        prep_data(args, confs)
    if args.lang == 'sq':
        confs = [
            wikiann_default_conf(args)
        ]
        prep_data(args, confs)
    return 0
