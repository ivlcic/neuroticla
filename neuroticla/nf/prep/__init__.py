import os
import logging
import pandas as pd

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

from .filter import DataFilter
from .aussda import AussdaLongDataFilter, AussdaShortDataFilter, AussdaManualDataFilter
from .slomcor import SlomcorDataFilter

logger = logging.getLogger('nf.prep')


def get_data_filter(arg) -> DataFilter:
    df: pd.DataFrame = pd.read_csv(
        arg.input_path,
        encoding='utf-8',
        nrows=10
    )
    logger.info("Got CVS columns after examine: %s", df.columns)
    if 'origin_ID' in df:
        return AussdaLongDataFilter(arg)
    elif 'ID_origin' in df:
        return AussdaShortDataFilter(arg)
    elif 'reminderid_doc_id' in df:
        return AussdaManualDataFilter(arg)
    else:
        return SlomcorDataFilter(arg)


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


def main(arg) -> int:
    logger.debug("Starting data preparation")
    if not os.path.exists(arg.input_file):
        arg.input_path = os.path.join(arg.data_in_dir, arg.input_file)

    df: DataFilter = get_data_filter(arg)
    df.load()
    df.filter()
    df.save()

    return 0
