import os
import logging

logger = logging.getLogger('ner.prep.wikiann')


def clean(args) -> None:
    data = ''
    invalid = {"''", "'", "]]", "[[", "==", "**", "``"}
    counter = 0
    sent_id = 0
    prefix = args.lang + ':'
    with open(os.path.join(args.process_file_name), 'rt', encoding='utf-8') as fp:
        line = 'whatever'
        while line:
            line = fp.readline()
            if line == '\n':
                if counter > 0:
                    data += '\n'
                counter = 0
                continue
            if line == '':
                continue
            tokens = line.split('\t')
            if tokens[0] and tokens[0].startswith(prefix):
                tokens[0] = tokens[0][len(prefix):]
            if tokens[0] in invalid:
                continue
            if tokens[0].startswith("''"):
                tokens[0] = tokens[0][2:]
            if tokens[0].startswith("**"):
                tokens[0] = tokens[0][2:]
            if counter == 0:
                data += '# sent_id = ' + str(sent_id) + '\n'
                sent_id += 1
            if counter == 0 and tokens[0] == '-':
                continue
            if counter == 0 and tokens[0] == '–':
                continue
            if counter == 0 and tokens[0] == ',':
                continue
            if counter == 0 and tokens[0] == ')':
                continue
            data += str(counter) + '\t' + tokens[0] + '\t' + tokens[1]
            counter += 1

    with open(os.path.join(args.process_file_name), 'wt', encoding='utf-8') as fp:
        fp.write(data)


def default_conf(args):
    conf = {
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
    return conf
