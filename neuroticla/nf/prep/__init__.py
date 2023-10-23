import os
import logging
import pandas as pd

from argparse import ArgumentParser

from ...core.args import CommonArguments
from ...core.split import DataSplit
from ...utils.zip import AESZipFile, ZIP_BZIP2, WZ_AES
from .filter import DataFilter
from .aussda import AussdaLongDataFilter, AussdaShortDataFilter, AussdaManualDataFilter
from .slomcor import SlomcorDataFilter, SlomcorManualDataFilter

logger = logging.getLogger('nf.prep')


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
    elif 'migration' in df:
        return SlomcorManualDataFilter(input_path, target_dir_path, base_name, num_rows)
    else:
        return SlomcorDataFilter(input_path, target_dir_path, base_name, num_rows)


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.processed_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    parser.add_argument('--num_rows', type=int, help='Numer of rows to use', default=None)
    parser.add_argument(
        '-p', '--password', type=str, help="Zip file password", required=True
    )
    parser.add_argument(
        'corpora', help='Corpora files (prefix) to prep.', nargs='+',
        choices=['aussda', 'slomcor']
    )


def main(arg) -> int:
    logger.debug("Starting data preparation")
    for corpora in arg.corpora:
        corpora_files = []
        for f in os.listdir(arg.data_in_dir):
            if not f.startswith(corpora):
                continue
            corpus_file = os.path.join(arg.data_in_dir, f)
            df: DataFilter = get_data_filter(corpus_file, arg.data_out_dir, corpora, arg.num_rows)
            df.load()
            df.filter()
            corpora_files.extend(df.save())
        if not corpora_files:
            continue
        zip_path = os.path.join(arg.data_out_dir, corpora + '.zip')
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with AESZipFile(
                zip_path, 'a', compression=ZIP_BZIP2, compresslevel=9
        ) as tmp_zip:
            tmp_zip.setencryption(WZ_AES, nbits=256)
            tmp_zip.setpassword(bytes(arg.password, encoding='utf-8'))  # intentional
            for f in corpora_files:
                tmp_zip.write(f, os.path.basename(f))
            tmp_zip.close()
        for f in corpora_files:
            os.remove(f)

    return 0
