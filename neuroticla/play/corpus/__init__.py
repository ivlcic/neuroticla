import logging

from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from . import dump, extract, stats, cluster
from ... import CommonArguments

logger = logging.getLogger('play.cluster')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.result_dir('corpus', parser, ('-o', '--result_dir'))
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
        '-c', '--country', help='Articles selection country.', type=str
    )
    parser.add_argument(
        'customers', help='Article selection comma-separated customers or file name.', type=str
    )


def corpus_dump(arg) -> int:
    return dump.dump(arg)


def corpus_correct_old(arg) -> int:
    return dump.correct_old(arg)


def corpus_correct(arg) -> int:
    return dump.correct(arg)


def corpus_extract(arg) -> int:
    return extract.extract_data(arg)


def corpus_sentiment(arg) -> int:
    return dump.sentiment(arg)


def corpus_sentiment2(arg) -> int:
    return dump.sentiment2(arg)


# ./play corpus stats_collect -s 2023-01-01 -e 2023-12-15 result/corpus/customers.txt
def corpus_stats_collect(arg) -> int:
    return stats.collect(arg)


# ./play corpus cluster_dump -c SI -s 2023-03-05 -e 2023-03-13 result/corpus/customers.txt
def corpus_cluster_dump(arg) -> int:
    return cluster.dump(arg)


def corpus_cluster_dump2(arg) -> int:
    return cluster.dump2(arg)
