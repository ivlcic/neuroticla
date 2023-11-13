import os
import logging
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from typing import List, Dict

from .e5 import e5_embed
from .utils import compare_clusterings, cluster_louvain, cluster_print
from .. import CommonArguments
from ..esdl import Elastika
from ..esdl.article import Article
from ..oai.embed import openai_embed

logger = logging.getLogger('play.cluster')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    # CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    beginning_of_day = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    parser.add_argument(
        '-s', '--start_date', help='Articles start selection date.', type=str,
        default=beginning_of_day.astimezone(timezone.utc).isoformat()
    )
    next_day = beginning_of_day + timedelta(days=1)
    parser.add_argument(
        '-e', '--end_date', help='Articles end selection date.', type=str,
        default=next_day.astimezone(timezone.utc).isoformat()
    )
    parser.add_argument(
        '-c', '--country', help='Articles selection country.', type=str,
        default='SI'
    )
    parser.add_argument(
        '-u', '--customer', help='Articles selection customer.', type=str,
        default='a65c7372-9fbe-410c-93d7-4613d26488e7'
    )


def cluster_compare(arg) -> int:
    arg.tmp_dir = os.path.join(arg.tmp_dir, 'cluster_articles')
    if not os.path.exists(arg.tmp_dir):
        os.makedirs(arg.tmp_dir)

    requests = Elastika()
    requests.filter_customer(arg.customer)
    requests.filter_country(arg.country)
    requests.field('vector_768___textonic_v1')

    articles: List[Article] = requests.gets(arg.start_date, arg.end_date)

    openai_embed(articles, 'oai_ada_002', arg.tmp_dir)
    e5_embed(articles, 'efed_e5', arg.tmp_dir)
    e5_embed(articles, 'e5', arg.tmp_dir)

    oai_l_clusters = cluster_louvain(articles, 'oai_ada_002', 0.92)
    fe5_l_clusters = cluster_louvain(articles, 'efed_e5', 0.95)
    e5_l_clusters = cluster_louvain(articles, 'e5', 0.91)
    ttnx_l_clusters = cluster_louvain(articles, 'vector_768___textonic_v1', 0.79)

    print('')
    print('========================== OpenAI ========================== ')
    cluster_print(oai_l_clusters)

    print('')
    print('========================= Feder E5 ========================= ')
    cluster_print(fe5_l_clusters)

    print('')
    print('==========================   E5   ========================== ')
    cluster_print(e5_l_clusters)

    print('')
    print('==========================   Textonic   ========================== ')
    cluster_print(ttnx_l_clusters)
    return 0
