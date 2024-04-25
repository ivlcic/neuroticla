import os
import logging
import numpy as np
import pandas as pd

from argparse import ArgumentParser

from .prep import DataFilter, AussdaLongDataFilter, AussdaShortDataFilter, AussdaManualDataFilter, SlomcorDataFilter
from .. import CommonArguments
from ..core.split import DataSplit

logger = logging.getLogger('nf.analyze')


def get_data_filter(input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> DataFilter:
    df: pd.DataFrame = DataSplit.read_csv(
        input_path,
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
        'corpora', help='Corpora files (or prefix) to analyze.'
        #, nargs='+',
        #choices=[
        #    'aussda', 'aussda_manual', 'aussda_short', 'slomcor', 'slomcor_middle_east', 'slomcor_ukraine'
        #]
    )


def _analyze(data, label_columns):
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

    grouped_df = data.agg(**agg_args).drop(['count'])
    grouped_df = pd.DataFrame([np.diag(grouped_df)], columns=grouped_df.columns)
    grouped_df.insert(0, 'sum', grouped_df.sum(axis=1))
    grouped_df.insert(0, 'count', data.shape[0])
    print(grouped_df.to_csv(sep='\t', index=False))

    print(label_columns)
    # Create a new column to store the label combinations
    data['comb'] = data[label_columns].apply(lambda row: ''.join(row.astype(str)), axis=1)
    # Calculate frequencies
    combination_freq = data['comb'].value_counts().reset_index()
    combination_freq.columns = ['comb', 'freq']
    # Calculate percentages
    combination_freq['rel_freq'] = (combination_freq['freq'] / data.shape[0])
    combination_freq = combination_freq.sort_values(by='comb').transpose()
    print(combination_freq.to_csv(sep='\t', index=False))


def analyze_raw(arg) -> int:
    logger.debug("Starting data preparation")
    for corpora in arg.corpora.split(','):
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

            _analyze(data, label_columns)

    return 0


def analyze_predicted(arg) -> int:
    input_file = os.path.join(arg.data_in_dir, arg.corpora)
    if not os.path.exists(input_file):
        input_file = arg.corpora
    if not os.path.exists(input_file):
        raise ValueError(f'Missing input file [{input_file}]')
    df: pd.DataFrame = DataSplit.read_csv(
        input_file,
        nrows=arg.num_rows
    )
    cols_with_prefix = [col for col in df.columns if col.startswith('p_')]
    _analyze(df, cols_with_prefix)
    cols_without_prefix = [col[2:] for col in cols_with_prefix]
    all_cols_exist = all(col in df.columns for col in cols_without_prefix)
    if all_cols_exist:
        _analyze(df, cols_without_prefix)
    return 0
