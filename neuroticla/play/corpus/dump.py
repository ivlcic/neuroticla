import os
import json
import logging

from datetime import datetime, timedelta
from sre_parse import State
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

from .__embed import _filter_write
from .__utils import load_range
from ...esdl import Elastika
from ...esdl.article import Article

logger = logging.getLogger('play.cluster.dump')


def dump(arg) -> int:
    model_name = 'intfloat/multilingual-e5-base'
    arg.tokenizer = AutoTokenizer.from_pretrained(model_name)
    arg.model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=os.path.join(arg.tmp_dir, model_name)
    )

    start_date = datetime.fromisoformat(arg.start_date)
    end_date = datetime.fromisoformat(arg.end_date)
    if os.path.exists(arg.customers):
        with open(arg.customers) as f:
            customers = f.read().splitlines()
    else:
        customers = arg.customers.split(',')
    logger.info("Dumping [%s::%s] for %s", start_date, end_date, customers)
    for customer in customers:
        requests = Elastika()
        requests.limit(9999)
        requests.filter_customer(customer)
        requests.field(['rubric', 'url'])

        current_date = end_date
        while current_date > start_date:
            prev_day = current_date - timedelta(days=1)
            day_dir = os.path.join(
                arg.result_dir, str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}"
            )
            articles: List[Article] = requests.get(prev_day, current_date)
            for a in articles:
                if not os.path.exists(os.path.join(day_dir, a.uuid + '.json')):
                    _filter_write(arg, a, day_dir)
            logger.info("Dumped [%s::%s] for [%s]", prev_day, current_date, customer)
            current_date = prev_day
    return 0


def correct_old(arg) -> int:
    start_date = datetime.fromisoformat(arg.start_date)
    end_date = datetime.fromisoformat(arg.end_date)
    if os.path.exists(arg.customers):
        with open(arg.customers) as f:
            customers = f.read().splitlines()
    else:
        customers = arg.customers.split(',')
    logger.info("Dumping [%s::%s] for %s", start_date, end_date, customers)
    for customer in customers:
        requests = Elastika()
        requests.limit(9999)
        requests.filter_customer(customer)
        requests.field(['rubric', 'url'])

        current_date = end_date
        while current_date > start_date:
            prev_day = current_date - timedelta(days=1)
            day_dir = os.path.join(
                arg.result_dir, str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}"
            )
            articles: List[Article] = requests.get(prev_day, current_date)
            for a in articles:
                article_file = os.path.join(day_dir, a.uuid + '.json')
                if not os.path.exists(article_file):
                    continue
                with open(article_file, encoding='utf-8') as json_file:
                    try:
                        saved_article = json.load(json_file)
                    except:
                        logger.error("Unable to load json file [%s] for [%s].", article_file, a)
                        os.remove(article_file)
                        return 1

                saved_article.pop('mediaReach', None)
                url = a.data.pop('url', None)
                if url:
                    saved_article['url'] = url

                saved_media = saved_article.get('media')
                media = a.data.get('media')
                if media and media.get('mediaReach', None):
                    saved_media['mediaReach'] = media.get('mediaReach')

                saved_rubric = saved_article.get('rubric')
                rubric = a.data.get('rubric')
                if rubric and rubric.get('mediaReach', None):
                    saved_rubric['mediaReach'] = rubric.get('mediaReach')

                with open(article_file, 'w', encoding='utf8') as json_file:
                    json.dump(saved_article, json_file, indent='  ', ensure_ascii=False)

                logger.info("Corrected [%s]", a)
            logger.info("Corrected [%s::%s] for [%s]", prev_day, current_date, customer)
            current_date = prev_day
    return 0


def correct(arg) -> int:
    def callback(s: State, saved: Dict[str, Any]) -> int:
        return 1

    state = load_range(arg.start_date, arg.end_date, arg.result_dir, callback)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    return 0
