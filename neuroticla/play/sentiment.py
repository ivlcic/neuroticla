import json
import logging
import os
import uuid

from argparse import ArgumentParser
from typing import List

import pandas as pd

from neuroticla import CommonArguments
from neuroticla.esdl import Elastika, Article
from neuroticla.play.corpus.__embed import _filter_body
from neuroticla.play.corpus.__utils import filter_article

logger = logging.getLogger('play.sentiment.dump')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.result_dir('corpus', parser, ('-o', '--result_dir'))
    parser.add_argument(
        'files', type=str
    )


# ./play sentiment dump /home/nikola/downloads/sentiment_export/serbian_rated_sentiment_export.csv,/home/nikola/downloads/sentiment_export/slovenian_sentiment_export.csv
def sentiment_dump(arg) -> int:
    files = arg.files.split(',')
    uuid_mapped_data = {}
    for file in files:
        if not os.path.exists(file):
            raise RuntimeError('file should be given to read sentiments!')
        dtype_dict = {
            'final_rate': float,
            'article_id': str,
            'named_entity': str
        }
        # data = pd.read_csv(file, delimiter=';', dtype=dtype_dict, nrows=10000)
        data = pd.read_csv(file, delimiter=';', dtype=dtype_dict)
        logger.info("Starting to parse %s", file)
        print(data.head())

        for index, row in data.iterrows():
            if row['article_id'] in uuid_mapped_data:
                article = uuid_mapped_data[row['article_id']]
                #logger.info("Duplicate sentiment in %s for %s.", file, row['article_id'])
            else:
                article = {
                    'uuid': row['article_id'],
                    'sentiments': []
                }
                uuid_mapped_data[row['article_id']] = article
            value = 0
            if row['final_rate'] <= -1:
                value = -1
            if row['final_rate'] >= 1:
                value = 1

            article['sentiments'].append({
                'source': 'analytics',
                'value': value,
                'aspect': row['named_entity']}
            )
        logger.info("Done with %s", file)

    def chunked_keys(sorted_keys, chunk_size):
        for i in range(0, len(sorted_keys), chunk_size):
            yield sorted_keys[i:i + chunk_size]

    # Sorting the dictionary by UUID (time-based sorting)
    sorted_keys = sorted(uuid_mapped_data.keys(), key=lambda x: uuid.UUID(x).time)
    print(len(sorted_keys))

    save_dir = os.path.join(arg.result_dir, 'sentiment')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    articles = []
    year = 1900
    for chunk in chunked_keys(sorted_keys, 500):
        esdl = Elastika()
        esdl.limit(1000)
        esdl.filter_uuid(chunk)
        esdl.field(['rubric', 'url', 'rates'])
        article_chunk: List[Article] = esdl.gets('1900-01-01', '2025-01-01')
        logger.info(
            'First [%s -- %s] and last [%s -- %s]',
            article_chunk[0].data['country']['name'],
            article_chunk[0].data['created'],
            article_chunk[-1].data['country']['name'],
            article_chunk[-1].data['created']
        )
        for a in article_chunk:
            filter_article(a)
            sentiments_item = uuid_mapped_data[a.uuid]
            body = ''
            filtered = False
            mt = a.data['media']['type']['name']
            if mt == 'radio' or mt == 'tv':
                for line_idx, line in enumerate(a.body.split('\n')):
                    tmp = _filter_body(line_idx, line)
                    if tmp:
                        body += '\n' if len(body) > 0 else ''
                        body += tmp
                    if not tmp and line:
                        filtered = True
            if not body:
                body = a.body
            body = body.replace(' ', ' ')
            if a.created.year != year:
                if len(articles) > 0:
                    file_name = os.path.join(save_dir, f'sent_analytics-{year}.json')
                    with open(file_name, 'w', encoding='utf8') as json_file:
                        json.dump(articles, json_file, indent='  ', ensure_ascii=False)
                articles = []
                year = a.created.year

            articles.append(
                {
                    'uuid': a.uuid,
                    #'created': a.data['created'],
                    'published': a.data['published'],
                    'language': a.language,
                    'country': a.country,
                    'sentiments': sentiments_item['sentiments'],
                    'title': a.title,
                    'body': body,
                    'ver': '1.1'
                }
            )

    file_name = os.path.join(save_dir, f'sent_analytics-{year}.json')
    with open(file_name, 'w', encoding='utf8') as json_file:
        json.dump(articles, json_file, indent='  ', ensure_ascii=False)
    return 0
