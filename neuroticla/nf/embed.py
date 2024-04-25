import os
import logging
import openai
import pandas as pd

from typing import List, Union
from argparse import ArgumentParser

from ..core.args import CommonArguments
from ..core.split import DataSplit
from ..oai.constants import EMBEDDING_ENCODING, EMBEDDING_CTX_LENGTH
from ..oai.tokenize import truncate_text_tokens
from ..utils.zip import AESZipFile, ZIP_BZIP2, WZ_AES

logger = logging.getLogger('nf.embed')


def openai_embed(data: pd.DataFrame, text_fields: Union[str, List[str]] = 'body'):
    # encode the text
    if isinstance(text_fields, str):
        text_fields = [text_fields]

    embeddings = []
    text_len = 0
    token_len = 0
    if 'embed_oai_ada2' in data:
        logger.info('Skipping OpenAI embedding column embed_oai_ada2 is already present.')
        return

    for i, row in data.iterrows():
        concatenated_string = '\n'.join([str(row[col]) for col in text_fields])
        tokens = truncate_text_tokens(
            concatenated_string,
            EMBEDDING_ENCODING,
            EMBEDDING_CTX_LENGTH
        )
        embedding = openai.Embedding.create(  # call OpenAI
            input=tokens, model="text-embedding-ada-002"
        )
        text_len += len(concatenated_string)
        token_len += len(tokens)
        logger.info('Loaded %s article OpenAI embedding.', i)
        embeddings.append(embedding["data"][0]["embedding"])
    data['embed_oai_ada2'] = embeddings
    logger.info(
        'Embedded %s articles of total text len %s and total token len %s with OpenAI embedding.',
        data.shape[0], text_len, token_len
    )


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.processed_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.split_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '-u', '--subsets', type=str, default=None,
        help='Subsets of the files to use for each corpora (file name contains any of the comma separated strings)',
    )
    parser.add_argument(
        '-p', '--password', type=str, required=True, help='Zip file password'
    )
    parser.add_argument('--num_rows', type=int, help='Numer of rows to use', default=None)
    parser.add_argument(
        '--openai', help="Do OpenAI ada-002 model embeddings", action='store_true'
    )
    parser.add_argument(
        'corpora', help='Corpora files to split.', nargs='+',
        choices=['aussda', 'slomcor']
    )


def main(arg) -> int:
    for corpus in arg.corpora:
        zip_path = os.path.join(arg.data_in_dir, corpus + '.zip')
        files = DataSplit.extract(
            zip_path,
            arg.password,
            arg.subsets.split(',') if arg.subsets else None,
            arg.data_out_dir
        )
        success = []
        for f in files:
            logger.info('Loaded file %s.', f)
            data = DataSplit.read_csv(f, nrows=arg.num_rows)
            if arg.openai:
                openai_embed(data, ['title', 'body'])
            # did we embed with something
            cols_with_prefix = [col for col in data.columns if col.startswith('embed_')]
            if arg.num_rows is None and len(cols_with_prefix) > 0:
                data.to_csv(f, encoding='utf-8', index=False)
                success.append(f)

        if len(success) > 0:
            with AESZipFile(
                    zip_path, 'a', compression=ZIP_BZIP2, compresslevel=9
            ) as tmp_zip:
                tmp_zip.setencryption(WZ_AES, nbits=256)
                tmp_zip.setpassword(bytes(arg.password, encoding='utf-8'))  # intentional
                for f in success:
                    tmp_zip.remove(os.path.basename(f))
                    tmp_zip.write(f, os.path.basename(f))
                tmp_zip.close()
    return 0
