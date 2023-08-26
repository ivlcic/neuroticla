import json
import logging
import os
import re
import shutil
from typing import Dict, List, Any

from ...core.labels import Labeler
from ...utils.zip import AESZipFile, ZIP_BZIP2, WZ_AES

logger = logging.getLogger('ner.prep.conll')


def to_conll(doc):
    if hasattr(doc, "to_conll"):
        return doc.to_conll()
    else:
        # noinspection PyStringFormat
        return "{:C}\n\n".format(doc)


def map_ner(map_filter: Dict, ner: str) -> str:
    mapped = map_filter.get(ner)
    if mapped is None:
        return ner
    else:
        return mapped


def add_tag_to_stats(stats: Dict, tag: str) -> None:
    ner_count = stats['tags'].get(tag)
    if ner_count is None:
        ner_count = 0
    stats['tags'][tag] = ner_count + 1
    if tag == 'O':
        return
    stats['tags']['NER'] = stats['tags']['NER'] + 1
    base_tag = tag.partition('-')[2]
    ner_count = stats['tags'].get(base_tag)
    if ner_count is None:
        ner_count = 0
    stats['tags'][base_tag] = ner_count + 1


def add_seq_to_stats(stats: Dict, seq: List, sentence_id: str, sentence: str) -> None:
    stats['num_sent'] = stats['num_sent'] + 1
    seq_len = len(seq)
    seq_len_cls: Dict = stats['seq_len_cls']
    if seq_len <= 32:
        seq_len_cls['32'] = seq_len_cls['32'] + 1
    elif seq_len <= 64:
        seq_len_cls['64'] = seq_len_cls['64'] + 1
    elif seq_len <= 128:
        seq_len_cls['128'] = seq_len_cls['128'] + 1
    elif seq_len <= 256:
        seq_len_cls['256'] = seq_len_cls['256'] + 1
        # logger.info('Found sequence longer than 128 tokens! Manual check needed at sentence id [%s][%s]!',
        #            sentence_id, sentence)
    elif seq_len <= 512:
        seq_len_cls['512'] = seq_len_cls['512'] + 1
        logger.info('Found sequence longer than 256 tokens! Manual check needed at sentence id [%s][%s]!',
                    sentence_id, sentence)
    else:
        logger.info('Found sequence longer than 512 tokens! Manual check needed at sentence id [%s][%s]!',
                    sentence_id, sentence)
    if seq_len > stats['longest_seq']:
        stats['longest_seq'] = seq_len


def parse_mwt_token(r: str) -> List[str]:
    mwt = r.split('-')
    result = []
    for x in range(int(mwt[0]), int(mwt[1]) + 1):
        result.append(str(x))
    return result


def to_csv(args, ner_tag_idx: int, map_filter: Dict[str, Any] = None):
    if map_filter is None:
        map_filter = {}

    labeler = Labeler(
        os.path.join(args.data_out_dir, 'tags.csv')
    )

    csv_fname = os.path.join(args.data_out_dir, args.target_base_name) + '.csv'
    json_fname = os.path.join(args.data_out_dir, args.target_base_name) + '.json'
    conll_fname = os.path.join(args.data_out_dir, args.target_base_name) + '.conll'
    csv = open(csv_fname, "a" if args.append else "w")
    csv.write('sentence,ner\n')
    logger.debug('Loading data [%s]', args.process_file_name)
    logger.debug('Reformatting data NER [%s -> %s]...', args.process_file_name, csv_fname)
    stats = {
        'tags': {
            'NER': 0, 'PER': 0, 'LOC': 0, 'ORG': 0, 'MISC': 0
        },
        'num_sent': 0,
        'longest_seq': 0,
        'seq_len_cls': {
            '32': 0, '64': 0, '128': 0, '256': 0, '512': 0
        }
    }
    stats['tags'].update(labeler.labels2ids())
    sentence = {'id': None, 'tokens': [], 'text': ''}
    max_seq_len = map_filter.get('max_seq_len', 128)
    stop_at = map_filter.get('stop_at', -1)
    with open(args.process_file_name) as fp:
        line = 'whatever'
        while line:
            line = fp.readline()
            if line.startswith('#'):
                # sent_id = train-s1
                if line.startswith("# sent_id = "):
                    sentence['id'] = line[12:].strip()
                # text = Proces privatizacije na Kosovu pod povećalom
                if line.startswith("# text = "):
                    sentence['text'] = line[9:].strip()
                continue
            if line == '\n' and sentence and sentence['tokens']:
                # process
                ner_tags = []
                sent_len = len(sentence['tokens'])
                if max_seq_len is not None and sent_len > max_seq_len:
                    logger.warning('Found sequence longer [%d] than [%d] tokens! Filtered out sentence id [%s][%s]!',
                                   sent_len, max_seq_len, sentence['id'], sentence['text'])
                    sentence = {'id': None, 'tokens': [], 'text': ''}
                    continue
                csv.write('"')
                mwt = []
                for token in sentence['tokens']:
                    if ner_tags:
                        csv.write(' ')
                    if token[1] == '"':
                        csv.write('""')
                    elif '"' in token[1]:
                        token[1].replace('"', '""')
                    else:
                        csv.write(token[1])
                    if "-" in token[0]:
                        mwt = parse_mwt_token(token[0])
                    if token[0] in mwt:
                        mwt.remove(token[0])
                        continue
                    ner_tag = token[ner_tag_idx]
                    if 'NER' in ner_tag:
                        ner_tag = re.findall(r'.*NER=([^|]+)', ner_tag)
                        if not ner_tag or len(ner_tag) <= 0:
                            logger.warning('Unable to parse NER tag at [%s:%s]', sentence['id'], sentence['text'])
                        ner_tag = ner_tag[0].strip()
                    if 'ner' in ner_tag:
                        ner_tag = re.findall(r'.*ner=([^|]+)', ner_tag)
                        if not ner_tag or len(ner_tag) <= 0:
                            logger.warning('Unable to parse ner tag at [%s:%s]', sentence['id'], sentence['text'])
                        ner_tag = ner_tag[0].strip()
                    if not ner_tag:
                        ner_tag = 'O'
                    ner_tag = ner_tag.strip()
                    ner_tag = map_ner(map_filter, ner_tag)
                    ner_tags.append(ner_tag)
                    add_tag_to_stats(stats, ner_tag)
                csv.write('",')
                csv.write(' '.join(ner_tags))
                csv.write("\n")
                add_seq_to_stats(stats, ner_tags, sentence['id'], sentence['text'])
                if 0 < stop_at <= stats['num_sent']:
                    logger.info("Forcefully stopping at sentence [%s]", stop_at)
                    break
                sentence = {'id': None, 'tokens': [], 'text': ''}
                continue
            data = line.split('\t')
            sentence['tokens'].append(data)

    logger.info('Reformatted data [%s -> %s]', args.process_file_name, csv_fname)
    logger.info('Reformatting stats: %s', stats)
    with open(json_fname, 'w') as outfile:
        json.dump(stats, outfile, indent=2)
    csv.close()
    shutil.copyfile(args.process_file_name, conll_fname)
    zip_path = os.path.join(args.data_out_dir, args.lang + '.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with AESZipFile(
            zip_path, 'a', compression=ZIP_BZIP2, compresslevel=9
    ) as myzip:
        myzip.setencryption(WZ_AES, nbits=256)
        myzip.setpassword(bytes(args.password, encoding='utf-8'))  # intentional
        myzip.write(conll_fname, args.target_base_name + '.conll')
        myzip.write(json_fname, args.target_base_name + '.json')
        myzip.write(csv_fname, args.target_base_name + '.csv')
        myzip.close()
    os.remove(conll_fname)
    os.remove(json_fname)
    os.remove(csv_fname)
