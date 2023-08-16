import os
import re
import logging
import pandas as pd

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

from .filter import DataFilter
from .aussda import AussdaLongDataFilter, AussdaShortDataFilter, AussdaManualDataFilter
from .slomcor import SlomcorDataFilter

logger = logging.getLogger('neuroticla.nf.prep')


def get_data_filter(args) -> DataFilter:
    df: pd.DataFrame = pd.read_csv(
        args.input_path,
        encoding='utf-8',
        nrows=10
    )
    if 'origin_ID' in df:
        return AussdaLongDataFilter(args)
    elif 'ID_origin' in df:
        return AussdaShortDataFilter(args)
    elif 'reminderid_doc_id' in df:
        return AussdaManualDataFilter(args)
    else:
        return SlomcorDataFilter(args)


def args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.processed_data_dir(nrcla_module, parser, ('-o', '--data_out_dir'))
    parser.add_argument('--num_rows', type=int, help='Numer of rows to use', default=None)
    parser.add_argument(
        'input_file',
        help='Corpora file (default: %(default)s)',
        type=str,
        default='articles_manual_annotated_925.csv'
    )


def main(args) -> int:
    logger.debug("Starting data preparation")
    if not os.path.exists(args.input_file):
        args.input_path = os.path.join(args.data_in_dir, args.input_file)

    df: DataFilter = get_data_filter(args)
    df.load()
    df.filter()
    df.save()
    logger.info("Got CVS columns after first filtering: %s", df.columns)

    return 0
