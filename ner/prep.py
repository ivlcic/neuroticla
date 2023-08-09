import logging
import os

from argparse import ArgumentParser
from neuroticla.args import CommonArguments

logger = logging.getLogger('ner.prep')


def check_param(conf: Dict, p_name: str) -> Any:
    p = conf.get(p_name)
    if not p:
        logger.warning('Missing [%s] param in [%s] config', p_name, conf)
        exit(1)
    return p


def check_dir_param(conf: Dict, param_name: str, parent_path: str) -> str:
    fname = chech_param(conf, param_name)
    fpath = os.path.join(parent_path, fname)
    if not os.path.exists(fpath):
        logger.warning('Missing [%s] filename in dir [%s]', fname, parent_path)
        exit(1)
    return fpath


def prep_data(args, confs: List[Dict]) -> None:
    for conf in confs:
        type = conf.get('type')
        if type == 'wikiann':
            ner_conll_idx = check_param(conf, 'ner_conll_idx')
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            lang = check_param(conf, 'lang')
            zip_fname = check_dir_param(conf, 'zip', args.corpora_dir)
            zip_dir = os.path.join(nf.default_tmp_dir, target_base_name)
            if not os.path.exists(zip_dir):
                os.mkdir(zip_dir)
            zipfile.ZipFile(zip_fname).extractall(zip_dir)
            proc_fname = check_dir_param(conf, 'proc_file', nf.default_tmp_dir)
            with open(os.path.join(proc_fname, 'train'), 'rt', encoding='utf-8') as fp:
                data = fp.read()
            with open(os.path.join(proc_fname, 'dev'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            with open(os.path.join(proc_fname, 'test'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            with open(os.path.join(proc_fname, 'extra'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            proc_fname = os.path.join(proc_fname, lang + '-wann.conll')
            with open(proc_fname, 'wt', encoding='utf-8') as fp:
                fp.write(data)
            filter_wikiann(lang, proc_fname)
            conll2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, ner_conll_idx, map_filter)
        if type == 'cnec':
            ner_conll_idx = check_param(conf, 'ner_conll_idx')
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            zip_fname = check_dir_param(conf, 'zip', args.corpora_dir)
            zipfile.ZipFile(zip_fname).extractall(nf.default_tmp_dir)
            proc_fname = check_dir_param(conf, 'proc_file', nf.default_tmp_dir)
            with open(os.path.join(proc_fname, 'dtest.conll'), 'rt', encoding='utf-8') as fp:
                data = fp.read()
            with open(os.path.join(proc_fname, 'train.conll'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            with open(os.path.join(proc_fname, 'etest.conll'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            proc_fname = os.path.join(proc_fname, 'cs-cnec.conll')
            with open(proc_fname, 'wt', encoding='utf-8') as fp:
                fp.write(data)
            filter_cnec(proc_fname)
            conll2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, ner_conll_idx, map_filter)
        if type == 'conll':
            zip_fname = check_dir_param(conf, 'zip', args.corpora_dir)
            zipfile.ZipFile(zip_fname).extractall(nf.default_tmp_dir)
            proc_fname = check_dir_param(conf, 'proc_file', nf.default_tmp_dir)
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            logger.debug('Converting conll data [%s -> %s]...', proc_fname, target_base_name)
            ner_conll_idx = check_param(conf, 'ner_conll_idx')
            conll2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, ner_conll_idx, map_filter)
            logger.info('Converted data [%s -> %s]', proc_fname, target_base_name)
        if type == 'bsnlp':
            zip_fname = check_dir_param(conf, 'zip', args.corpora_dir)
            zipfile.ZipFile(zip_fname).extractall(nf.default_tmp_dir)
            proc_fname = check_dir_param(conf, 'proc_file', nf.default_tmp_dir)
            target_base_name = check_param(conf, 'result_name')
            map_filter = check_param(conf, 'map_filter')
            logger.debug('Converting BSNLP data [%s -> %s]...', proc_fname, target_base_name)
            bsnlp2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, map_filter)
            logger.info('Converted data [%s -> %s]', proc_fname, target_base_name)


def args(package: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(package, parser, ('-i', '--data_in_dir'))
    CommonArguments.processed_data_dir(package, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(package, parser, ('-t', '--tmp_dir'))
    parser.add_argument('lang', help='language of the text',
                        choices=['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'bg', 'pl', 'ru', 'sk', 'uk'])


def main(args) -> int:
    if args.lang == 'sl':
        tokenizer = nf.data.get_classla_tokenizer(args.lang)
        confs = [
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'ssj500k-syn.ud.conllu'),
                'result_name': args.lang + '_500k',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 128,
                    'stop_at': 9483,
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
                    'max_seq_len': 128,
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
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'hr':
        tokenizer = nf.data.get_classla_tokenizer(args.lang)
        confs = [
            {
                'type': 'conll',
                'zip': 'hr500k-1.0.zip',
                'proc_file': os.path.join('hr500k.conll', 'hr500k.conll'),
                'result_name': args.lang + '_500k',
                'ner_conll_idx': 10,
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'sr':
        confs = [
            {
                'type': 'conll',
                'zip': 'setimes-sr.conll.zip',
                'proc_file': os.path.join('setimes-sr.conll', 'set.sr.conll'),
                'result_name': args.lang + '_set',
                'ner_conll_idx': 10,
                'map_filter': {
                    'max_seq_len': 128,
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
        nf.data.multi_split_data(args, confs)
    if args.lang == 'cs':
        tokenizer = nf.data.get_stanza_tokenizer(args.lang)
        confs = [
            {
                'type': 'cnec',
                'zip': 'CNEC_2.0_konkol.zip',
                'proc_file': 'CNEC_2.0_konkol',
                'result_name': args.lang + '_cnec',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128,
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
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'bg':
        tokenizer = nf.data.get_classla_tokenizer(args.lang)
        confs = [
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'ru':
        tokenizer = nf.data.get_stanza_tokenizer(args.lang)
        confs = [
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'pl':
        tokenizer = nf.data.get_stanza_tokenizer(args.lang)
        confs = [
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'sk':
        tokenizer = nf.data.get_stanza_tokenizer(args.lang)
        confs = [
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            },
            {
                'type': 'wikiann',
                'lang': args.lang,
                'zip': args.lang + '-wann.zip',
                'proc_file': args.lang + '_wann',
                'result_name': args.lang + '_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'uk':
        tokenizer = nf.data.get_stanza_tokenizer(args.lang)
        confs = [
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': args.lang + '_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': args.lang,
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'bs':
        confs = [
            {
                'type': 'wikiann',
                'lang': args.lang,
                'zip': args.lang + '-wann.zip',
                'proc_file': args.lang + '_wann',
                'result_name': args.lang + '_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'mk':
        confs = [
            {
                'type': 'wikiann',
                'lang': args.lang,
                'zip': args.lang + '-wann.zip',
                'proc_file': args.lang + '_wann',
                'result_name': args.lang + '_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    if args.lang == 'sq':
        confs = [
            {
                'type': 'wikiann',
                'lang': args.lang,
                'zip': args.lang + '-wann.zip',
                'proc_file': args.lang + '_wann',
                'result_name': args.lang + '_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        nf.data.multi_split_data(args, confs)
    return 0
