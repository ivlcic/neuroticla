import os
import json
import logging
import zoneinfo

from datetime import datetime, timedelta, timezone
from typing import Callable, Any, Dict, List, Union

from transformers import PreTrainedTokenizer, PreTrainedModel

from ...esdl import Elastika, Article

logger = logging.getLogger('play.cluster.dump')


class State:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end
        self.currentDate: Union[datetime, None] = None
        self.prevDate: Union[datetime, None] = None
        self.index: int = 0
        self.size: int = 0
        self.total: int = 0
        self.file: str = ''
        self.relPath: str = ''
        self.relDir: str = ''
        self.log: Dict[str, Any] = {}


tag_callback = Callable[[int, Dict[str, Any]], None]


class Params:
    def __init__(self, start_date: str, end_date: str, customers: Union[None, List[str]], result_dir: str):
        self.start_date = start_date
        self.end_date = end_date
        self.start = datetime.fromisoformat(start_date).astimezone()
        self.end = datetime.fromisoformat(end_date).astimezone()
        self.result_dir = result_dir
        self.customers = customers
        # self.customersCtg = [str(uuid3(uuid.NAMESPACE_URL, 'CustomerTopicGroup.' + x)) for x in customers]
        self.requests: Union[Elastika, None] = None
        self.tokenizer: Union[PreTrainedTokenizer, None] = None
        self.model: Union[PreTrainedModel, None] = None
        self.lang: Union[str, None] = None
        self.skipEmbedding: bool = False
        self.tagCallback: Union[tag_callback, None] = None

    def set_requests(self, requests: Elastika):
        self.requests = requests

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def set_model(self, model: PreTrainedModel):
        self.model = model


load_range_callback = Callable[[State, Dict[str, Any], Union[Article, None]], int]


def filter_article(article: Article):
    article.data.pop('advertValue', None)
    article.data.pop('mediaReach', None)
    article.data.pop('translations', None)
    url = article.data.pop('url', None)
    if url:
        article.data['url'] = url

    media = article.data.get('media')
    media.pop('publisher', None)
    media.pop('advertValue', None)
    media.pop('circulation', None)
    media.pop('providerId', None)
    media.pop('descriptor', None)
    media.pop('class', None)
    media.pop('language', None)
    tags = media.pop('tags', [])
    tags[:] = [tup for tup in tags if tup.get('class', '') == 'org.dropchop.jop.beans.tags.MediaType']
    media['type'] = {
        'name': tags[0]['name'],
        'uuid': tags[0]['uuid']
    }

    rubric = article.data.get('rubric')
    if rubric:
        rubric.pop('advertValue', None)
        rubric.pop('providerId', None)
        rubric.pop('descriptor', None)
        rubric.pop('class', None)

    country = article.data.get('country')
    country.pop('tags', None)
    country.pop('descriptor', None)
    country.pop('class', None)

    for k, v in article.data.get('translations', {}).items():
        v.pop('bodyPages')
        v.pop('bodyOctetLen')
        v.pop('bodyCalculatedPages')
        v.pop('bodyMd5')
        v.pop('bodyBillingPages')
        v.pop('class')
        v.pop('uuid')

    tags = article.data.pop('tags', [])
    article.data['topic_names'] = {}
    tags[:] = [tup for tup in tags if tup.get('class', '') != 'org.dropchop.jop.beans.tags.Genre']
    for i, t in enumerate(tags):
        t.pop('descriptor', None)
        tag_name = t.pop('name', None)
        t.pop('_OOP', None)
        if t.pop('class', '') == 'org.dropchop.jop.beans.tags.CustomerTopic':
            t['type'] = 'topic'
            if tag_name:
                article.data['topic_names'][t['uuid']] = tag_name
                t['name'] = tag_name
        if t.pop('class', '') == 'org.dropchop.jop.beans.tags.CustomerTopicGroup':
            t['type'] = 'group'

    article.data['tags'] = tags
    # logger.info("Done filtering [%s]", article)


def _filter_tags(params: Params, level: int, sub_dict: Dict[str, Any]) -> bool:
    # if level > 0:
    #     if level > 1:
    #         return False
    #     if 'refUuid' in sub_dict:
    #         sub_dict['uuid'] = sub_dict.pop('refUuid', None)
    #         sub_dict['type'] = 'topic_group'
    #     if params.tagCallback:
    #         params.tagCallback(level, sub_dict)
    #     return True

    if 'type' in sub_dict and sub_dict['type'] == 'topic':
        # for ctg in sub_dict['tags']:
        #    if 'uuid' in ctg and ctg['uuid'] not in params.customersCtg:
        #        return False
        for ctg in sub_dict['tags']:
            if 'refUuid' in ctg:
                sub_dict['parent'] = ctg['refUuid']
            if 'uuid' in ctg:
                sub_dict['parent'] = ctg['uuid']
        if 'parent' in sub_dict:
            sub_dict.pop('tags', None)
        if params.tagCallback:
            params.tagCallback(level, sub_dict)
        return True
    else:
        return False


def traverse_article_tags(params: Params, level: int, d: Dict[str, Any]):
    if not ('tags' in d and isinstance(d['tags'], list)):
        return
    for i in range(len(d['tags']) - 1, -1, -1):
        if not _filter_tags(params, level, d['tags'][i]):
            del d['tags'][i]
            continue
        # traverse_article_tags(params, level + 1, d['tags'][i])
    if not d['tags']:
        d.pop('tags', None)


def load_range(params: Params, callback: load_range_callback) -> State:
    state: State = State(params.start, params.end)
    if not params.customers:
        return fs_range(params, callback)
    for customer in params.customers:
        params.requests = Elastika()
        params.requests.limit(9999)
        params.requests.filter_customer(customer)
        params.requests.field(['rubric', 'url', 'rates'])
        s = fs_range(params, callback)
        state.currentDate = s.currentDate
        state.index += s.index
        state.total += s.total
    logger.info("Dumped [%s::%s] for %s", params.start, params.end, params.customers)
    return state


def fs_range(params: Params, callback: load_range_callback) -> State:
    state: State = State(params.start, params.end)
    current_date = params.end
    while current_date > params.start:
        prev_day = current_date - timedelta(days=1)
        rel_path = os.path.join(str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}")
        state.relDir = rel_path
        day_dir = os.path.join(params.result_dir, rel_path)
        if not os.path.exists(day_dir):
            os.makedirs(day_dir)

        state.currentDate = current_date
        state.prevDate = prev_day

        if params.requests:  # load article from Elasticsearch
            articles: List[Article] = params.requests.get(prev_day, current_date)
            state.size = len(articles)
            for x, a in enumerate(articles):
                state.file = os.path.join(day_dir, a.uuid + '.json')
                saved_article = {}
                state.article = saved_article
                if os.path.exists(state.file):
                    with open(state.file, encoding='utf-8') as json_file:
                        # noinspection PyBroadException
                        try:
                            saved_article = json.load(json_file)
                        except Exception:
                            logger.error("Unable to load json file [%s].", state.file)
                            os.remove(state.file)
                            continue
                state.index = x
                state.relPath = day_dir
                state.total += callback(state, saved_article, a)
        else:  # load article from filesystem
            file_names = os.listdir(day_dir)
            state.size = len(file_names)
            for x, article_file in enumerate(file_names):
                if not article_file.endswith('.json'):
                    continue
                article_file = os.path.join(day_dir, article_file)
                if not os.path.exists(article_file):
                    continue
                with open(article_file, encoding='utf-8') as json_file:
                    # noinspection PyBroadException
                    try:
                        saved_article = json.load(json_file)
                    except Exception:
                        logger.error("Unable to load json file [%s].", article_file)
                        os.remove(article_file)
                        return state
                    state.index = x
                    state.article = saved_article
                    state.file = article_file
                    state.relPath = day_dir
                    state.total += callback(state, saved_article, None)

        logger.info("Finished interval [%s::%s] %s", prev_day, current_date, state.log)
        current_date = prev_day
    return state
