import os
import re
import logging
import pandas as pd

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

logger = logging.getLogger('neuroticla.nf.prep')

def cleanup_text(s: str):
    if not isinstance(s, str):
        return
    s = s.replace(' ', ' ').replace('\t', ' ').replace('\r', '')
    s = re.sub('"+', '"', s)
    s = re.sub(' +\n+', '\n', s)
    s = re.sub('\n+', '\n', s)
    s = re.sub(' +', ' ', s)
    return s.strip()

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

    df = pd.read_csv(
        args.input_path,
        dtype={
            'headline': 'string',
            'text': 'string',
            'lead_paragraph': 'string',
            'country': 'string',
            'source': 'string',
            'publication_date': 'string',
            "ID": int,
            "ID_aussda": 'Int64',
            "ID_origin": 'Int64',
            "ID_sample": 'Int64',
            "fr_eco": 'Int64',
            'fr_lab': 'Int64',
            'fr_wel': 'Int64',
            'fr_sec': 'Int64',
            'fr_cul': 'Int64',
            'middle_east': 'Int64',
            'eastern_europe': 'Int64'
        },
        encoding='utf-8',
        nrows=args.num_rows
    )
    logger.info("Got CVS columns: %s", df.columns)
    df.rename(columns={
        "ID": "id",
        "publication_date": "published",
        'headline': 'title',
        # 'headline_mt': 'title',
        'lead_paragraph': 'lead',
        # 'lead_paragraph_mt': 'lead',
        'text': 'body',
        # 'text_mt': 'body'
    }, inplace=True)
    columns = [
        'fr_eco', 'fr_lab', 'fr_wel', 'fr_sec', 'fr_cul', 'middle_east', 'eastern_europe',
        'headline', 'text', 'lead_paragraph'
    ]
    df = df.dropna(
        subset=columns
    )
    df = df[[
        'id', 'country', 'source', 'published',
        'fr_eco', 'fr_lab', 'fr_wel', 'fr_sec', 'fr_cul', 'middle_east', 'eastern_europe',
        'title', 'lead', 'body'
    ]]
    logger.info("Got CVS columns after first filtering: %s", df.columns)

    for i, row in df.iterrows():
        body: str = cleanup_text(df.at[i, 'body'])
        title: str = cleanup_text(df.at[i, 'title'])
        lead: str = cleanup_text(df.at[i, 'lead'])
        if body.startswith(title):
            body = body[len(title):]
        df.at[i, 'body'] = body
        df.at[i, 'title'] = title
        df.at[i, 'lead'] = lead

    return 0
