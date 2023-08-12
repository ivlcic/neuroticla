import os
import re
import logging

from typing import Dict, Any, Callable
from io import StringIO

from .conll import to_csv as conll_to_csv
from .conll import to_conll
from .tokens import get_obeliks_tokenizer, get_reldi_tokenizer, get_stanza_tokenizer

logger = logging.getLogger('ner.prep.bsnlp')


def read_file_skip_lines(path: str, skip: int, line_clbk: Callable = None) -> str:
    file1 = open(path, 'r')
    buf = StringIO()
    count = 0
    while True:
        count += 1
        line = file1.readline()
        if not line:
            break
        if count <= skip:
            continue
        if line_clbk is not None:
            line_clbk(count - 1, line.strip())
        buf.write(line)
    return buf.getvalue()


def bsnlp_init_anno_record(record: Dict, map_filter: Dict[str, Any]):
    ner_tags = []

    def process_ner_line(idx: int, line: str):
        data = line.split('\t')
        tag = data[2]
        form = data[0]
        tokens = map_filter['tokenizer'](form).sentences[0].tokens
        lower = True
        for t in tokens:
            if t.text.isalpha() and not t.text.islower():
                lower = False
                break

        ner_tags.append({
            'tag': tag,
            'form': form,
            'tokens': [x.text for x in tokens],
            'lower': lower,
            't_len': len(tokens),
            'sort': len(tokens) * 10000 + len(form)
        })

    read_file_skip_lines(record['a_fname'], 1, process_ner_line)
    ner_tags.sort(key=lambda x: x['sort'], reverse=True)
    record['a_ner_t'] = ner_tags


def bsnlp_process_raw_record(record: Dict, map_filter: Dict):
    record['r_text'] = read_file_skip_lines(record['r_fname'], 4)
    idx = 0
    text_token_list = []
    token_list = []
    doc = map_filter['tokenizer'](record['r_text'])
    lowerc = True
    for ner_tag in record['a_ner_t']:
        if not ner_tag['lower']:
            lowerc = False
            break

    for sent in doc.sentences:
        for sent_tok in sent.tokens:
            idx = record['r_text'].index(sent_tok.text, idx)
            sent_tok._start_char = idx
            sent_tok._ner = 'O'
            sent_tok._end_char = idx + len(sent_tok.text) - 1
            token_list.append(sent_tok)
            if lowerc:
                text_token_list.append(sent_tok.text.lower())
            else:
                text_token_list.append(sent_tok.text)
            idx = sent_tok.end_char + 1

    count = 0
    for ner_tag in record['a_ner_t']:
        ner_t_len = ner_tag['t_len']
        for idx in range(len(text_token_list) - ner_t_len + 1):
            if text_token_list[idx: idx + ner_t_len] == ner_tag['tokens']:
                for j in range(idx, idx + ner_t_len):
                    if token_list[j]._ner != 'O':
                        break
                    if j == idx:
                        token_list[j]._ner = 'B-' + ner_tag['tag']
                        count = + 1
                    else:
                        token_list[j]._ner = 'I-' + ner_tag['tag']
    if count == 0 and len(record['a_ner_t']) > 0:
        logger.warning('No NER matched in [%s] with annotations in [%s]!', record['r_fname'], record['a_fname'])
    record['conll'] = '# new_doc_id = ' + record['topic'] + '-' + record['id'] + '\n' + to_conll(doc)


def bsnlp_create_records(bsnlp_path: str, lang: str, map_filter: Dict = None) -> Dict[str, Dict[str, Any]]:
    if map_filter is None:
        map_filter = {}
    ignore_dirs = map_filter.get('ignore_dirs')
    if ignore_dirs is None:
        ignore_dirs = []
    anno_path = os.path.join(bsnlp_path, 'annotated')
    raw_path = os.path.join(bsnlp_path, 'raw')
    anno_records = {}
    _, dirs, _ = next(x for x in os.walk(anno_path))
    dirs.sort()
    for d in dirs:
        if d in ignore_dirs:
            continue
        logger.debug('Processing directory [%s]', d)
        a_lang_path = os.path.join(anno_path, d, lang)
        r_lang_path = os.path.join(raw_path, d, lang)
        if os.path.exists(a_lang_path) and os.path.exists(r_lang_path):
            _, _, a_files = next(x for x in os.walk(a_lang_path))
            _, _, r_files = next(x for x in os.walk(r_lang_path))
            a_files.sort()
            r_files.sort()
            for af in a_files:
                afnum = re.findall(r'(\d+([-_]\d+)*)', af)
                if not afnum or len(afnum) <= 0 or len(afnum[0]) <= 0:
                    logger.warning('Unable to parse id from annotated file [%s]', af)
                    continue
                a_fname = os.path.join(a_lang_path, af)
                record = {
                    'a_fname': a_fname,
                    'topic': d,
                    'id': afnum[0][0]
                }
                bsnlp_init_anno_record(record, map_filter)
                anno_records[d + '-' + afnum[0][0]] = record
            for rf in r_files:
                # rfnum = re.findall(r'\d+', rf)
                rfnum = re.findall(r'(\d+([-_]\d+)*)', rf)
                if not rfnum or len(rfnum) <= 0 or len(rfnum[0]) <= 0:
                    logger.warning('Unable to parse id from raw file [%s]', rf)
                    continue
                record = anno_records.get(d + '-' + rfnum[0][0])
                if record is None:
                    logger.warning('Unable to find matching annotated file [%s]', rf)
                    continue
                record['r_fname'] = os.path.join(r_lang_path, rf)
                bsnlp_process_raw_record(record, map_filter)

    return anno_records


def to_csv(args, map_filter: Dict = None):
    if map_filter is None:
        map_filter = {}
    conll_fname = os.path.join('tmp', 'out.conll')
    anno_records = bsnlp_create_records(args.process_file_name, args.lang, map_filter)
    conll_fp = open(conll_fname, "a" if args.append else "w")
    for k, record in anno_records.items():
        conll_fp.write(record['conll'])
    conll_fp.close()
    logger.info('Reformatted data [%s -> %s]', args.process_file_name, conll_fname)
    args.process_file_name = conll_fname
    conll_to_csv(args, 9, map_filter)


def default_conf(args):
    obeliks_set = {'sl'}
    reldi_set = {'hr', 'sr', 'bs', 'mk', 'bg'}
    if args.lang in obeliks_set:
        tokenizer = get_obeliks_tokenizer(args)
    elif args.lang in reldi_set:
        tokenizer = get_reldi_tokenizer(args)
    else:
        tokenizer = get_stanza_tokenizer(args)
    conf = {
        'type': 'bsnlp',
        'zip': 'bsnlp-2017-21.zip',
        'proc_file': 'bsnlp',
        'result_name': args.lang + '_bsnlp',
        'map_filter': {
            'max_seq_len': 256,
            'lang': args.lang,
            'tokenizer': tokenizer,
            'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
            'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
        }
    }
    return conf
