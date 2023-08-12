import os
import logging

logger = logging.getLogger('ner.prep.conll')


def clean(args):
    # I = ORG
    # T = MISC (time) -> O
    # P = PER
    # G = LOC
    # M = MISC (media) -> ORG (if not @)
    # O = MISC (product)
    # A = MISC (address, number) -> O
    # B-[^ITPGMAO]{1,}$
    data = ''
    counter = 0
    sent_id = 0
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
            if counter == 0:
                data += '# sent_id = ' + str(sent_id) + '\n'
                sent_id += 1
            if '@' in tokens[0] and tokens[3].endswith('M'):
                tokens[3] = 'O'
            data += str(counter) + '\t' + tokens[0] + '\t' + tokens[3]
            counter += 1

    with open(os.path.join(args.process_file_name), 'wt', encoding='utf-8') as fp:
        fp.write(data)
