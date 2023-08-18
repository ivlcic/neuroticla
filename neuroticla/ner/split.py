import os
import shutil
import logging

import neuroticla.utils.zip

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments
from neuroticla.core.split import DataSplit

logger = logging.getLogger('ner.split')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.processed_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.split_data_dir(nrcla_module, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(nrcla_module, parser, ('-t', '--tmp_dir'))
    CommonArguments.data_split(parser)
    parser.add_argument(
        '-u', '--subsets', type=str, default=None,
        help="Subsets of the files to use for each language (file name contains any of the comma separated strings)",
    )
    parser.add_argument(
        '-p', '--password', type=str, default='showeffort',
        help="Zip file password",
    )
    parser.add_argument(
        '-r', '--non_reproducible_shuffle', action='store_true', default=False,
        help='Non reproducible data shuffle.',
    )
    parser.add_argument(
        'langs', help='languages to split', nargs='+',
        choices=['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'bg', 'pl', 'ru', 'sk', 'uk']
    )


def main(arg) -> int:
    logger.debug("main")
    files = {}
    for lang in arg.langs:
        lang_dir_path = os.path.join(arg.data_out_dir, lang)
        if not os.path.exists(lang_dir_path):
            os.makedirs(lang_dir_path)
        lang_zip_path = os.path.join(arg.data_in_dir, lang + '.zip')
        with neuroticla.utils.zip.AESZipFile(
                lang_zip_path, 'a',
                compression=neuroticla.utils.zip.ZIP_BZIP2,
                compresslevel=9
        ) as myzip:
            myzip.setencryption(neuroticla.utils.zip.WZ_AES, nbits=256)
            myzip.setpassword(bytes(arg.password, encoding='utf-8'))
            myzip.extractall(lang_dir_path)
        files[lang] = []
        for f in os.listdir(lang_dir_path):
            if not f.endswith('.csv'):
                continue
            if arg.subsets is not None:
                for s in arg.subsets.split(','):
                    if s in f:
                        files[lang].append(os.path.join(lang_dir_path, f))
            else:
                files[lang].append(os.path.join(lang_dir_path, f))

    if arg.non_reproducible_shuffle:
        training_data, evaluation_data, test_data = DataSplit.multi_split(arg.data_split, files)
    else:
        training_data, evaluation_data, test_data = DataSplit.multi_split(arg.data_split, files, 2611)

    target_path = os.path.join(arg.data_out_dir, '_'.join(arg.langs))
    training_data.to_csv(
        target_path + '.train.csv', index=False, encoding='utf-8'
    )
    evaluation_data.to_csv(
        target_path + '.eval.csv', index=False, encoding='utf-8'
    )
    test_data.to_csv(
        target_path + '.test.csv', index=False, encoding='utf-8'
    )
    for lang in arg.langs:
        lang_dir_path = os.path.join(arg.data_out_dir, lang)
        shutil.rmtree(lang_dir_path)
    return 0
