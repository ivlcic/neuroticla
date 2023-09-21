import os
import json
import csv
import logging
import pandas as pd

from argparse import ArgumentParser
from .. import CommonArguments

logger = logging.getLogger('nf.convert')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.result_dir(module_name, parser, ('-i', '--data_in_dir'))
    parser.add_argument(
        'input_file', help='Result json file.'
    )


def flatten(current, key, result):
    if isinstance(current, dict):
        for k in current:
            new_key = "{0}.{1}".format(key, k) if len(key) > 0 else k
            flatten(current[k], new_key, result)
    else:
        result[key] = current
    return result


def main(arg) -> int:
    logger.debug("Starting json results conversion")
    input_file = os.path.join(arg.data_in_dir, arg.input_file)
    if not os.path.exists(input_file):
        input_file = arg.input_file
    if not os.path.exists(input_file):
        raise ValueError(f'Missing input file [{input_file}]')
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Opening JSON file and loading the data
    # into the variable data
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    results = []
    for model_name, model_data in data.items():
        model_data['model_name'] = model_name
        model_result = flatten(model_data, '', {})
        print(model_result)
        results.append(model_result)

    d = pd.DataFrame(results)
    d.to_csv(os.path.join(arg.data_in_dir, base_name + '.csv'))
    return 0
