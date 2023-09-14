import os
import logging
import pandas as pd

from argparse import ArgumentParser
from typing import List

from .prep import DataFilter, AussdaLongDataFilter, AussdaShortDataFilter, AussdaManualDataFilter, SlomcorDataFilter
from .utils import get_data_path_prefix
from .. import CommonArguments

logger = logging.getLogger('nf.analyze')


def get_data_filter(input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> DataFilter:
    df: pd.DataFrame = pd.read_csv(
        input_path,
        encoding='utf-8',
        nrows=10
    )
    logger.info("Got CVS columns after examine: %s", df.columns)
    if 'origin_ID' in df:
        return AussdaLongDataFilter(input_path, target_dir_path, base_name, num_rows)
    elif 'ID_origin' in df:
        return AussdaShortDataFilter(input_path, target_dir_path, base_name, num_rows)
    elif 'reminderid_doc_id' in df:
        return AussdaManualDataFilter(input_path, target_dir_path, base_name, num_rows)
    else:
        return SlomcorDataFilter(input_path, target_dir_path, base_name, num_rows)


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument('--num_rows', type=int, help='Numer of rows to use', default=None)
    parser.add_argument(
        'corpora', help='Corpora files (prefix) to prep.', nargs='+',
        choices=[
            'aussda', 'aussda_manual', 'aussda_short', 'slomcor', 'slomcor_middle_east', 'slomcor_ukraine'
        ]
    )


def main(arg) -> int:
    logger.debug("Starting data preparation")
    for corpora in arg.corpora:
        for f in os.listdir(arg.data_in_dir):
            if not f.startswith(corpora):
                continue
            corpus_file = os.path.join(arg.data_in_dir, f)
            df: DataFilter = get_data_filter(corpus_file, arg.tmp_dir, corpora, arg.num_rows)
            df.load()
            data = df.data()
            label_columns = df.label_cols()
            if not label_columns:
                continue

            agg_args = {
                'count': pd.NamedAgg(column=label_columns[0], aggfunc='count')
            }
            for l in label_columns:
                agg_args[l] = pd.NamedAgg(column=l, aggfunc='sum')
            grouped_df = data.groupby(['country', 'source']).agg(
                **agg_args
            ).reset_index()
            print(grouped_df.to_csv(sep='\t', index=None))

            grouped_df = data.groupby(['country']).agg(
                **agg_args
            ).reset_index()
            print(grouped_df.to_csv(sep='\t', index=None))

            print(label_columns)
            # Create a new column to store the label combinations
            data['comb'] = data[label_columns].apply(lambda row: ''.join(row.astype(str)), axis=1)
            # Calculate frequencies
            combination_freq = data['comb'].value_counts().reset_index()
            combination_freq.columns = ['comb', 'freq']
            # Calculate percentages
            combination_freq['rel_freq'] = (combination_freq['freq'] / data.shape[0])
            print(combination_freq.to_csv(sep='\t', index=None))

    return 0
