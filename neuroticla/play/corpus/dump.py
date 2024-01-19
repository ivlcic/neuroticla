import os
import csv
import json
import logging

from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable

import requests
from transformers import AutoTokenizer, AutoModel

from .__embed import _filter_write
from .__utils import load_range, State
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


def _get_keywords(topic_uuds: List[str], uuid, title, content) -> Dict[str, Any]:
    req_data = {
        "provider": "admin",
        "articleUuid": uuid,
        "title": title,
        "content": content,
        "includeTopics": topic_uuds,
        "principalCheck": False,
        "runMediaRules": False,
        "activeKeywords": False
    }
    req_data_str = json.dumps(req_data, indent=2)
    try:
        # make HTTP verb parameter case-insensitive by converting to lower()
        resp = requests.post('https://stag-admin.klipingmap.com/cxf/Resolver/resolveTopicsAndEntities',
                             headers={
                                 'Content-Type': 'application/json',
                                 'X-Client-Uuid': 'cac6d703-a941-4fc0-bb23-1201a5976718'
                             },
                             data=req_data_str)
    except Exception as error:
        logger.error('Resolve request [%s] error [%s]:', req_data_str, error)
        raise error
    try:
        resp_text: Dict[str, Any] = json.loads(resp.text)
        return resp_text
    except Exception as error:
        logger.error('Elasticsearch parse error [%s]:', resp.text)
        raise error


def _traverse_tags(level: int, d: Dict[str, Any], clbk_filter: Callable[[int, Dict[str, Any]], bool]):
    if not ('tags' in d and isinstance(d['tags'], list)):
        return
    for i in range(len(d['tags']) - 1, -1, -1):
        if not clbk_filter(level, d['tags'][i]):
            del d['tags'][i]
            continue
        _traverse_tags(level + 1, d['tags'][i], clbk_filter)


def _extract_intervals(matched: List[Dict[str, Any]], kwe_iptc_map: Dict[str, Dict[str, str]]):
    results = {
        'kwe': {},
        'spans': []
    }
    for m in matched:
        result = {'start': m['start'], 'end': m['end'], 'kwe': [], 'tags': []}
        results['spans'].append(result)
        for k in m['matches']:
            k_uuid = k['category']['id']
            t_uuid = k['category']['metadata']['topicId']
            result['kwe'].append(k_uuid)
            result['tags'].append(t_uuid)
            if k_uuid not in results['kwe']:
                kwe = {
                    'topic_uuid': t_uuid,
                    'topic_name': k['category']['metadata']['topicName'],
                    'expr': k['category']['metadata']['regex']
                }
                # if k_uuid in kwe_iptc_map: #  better to map this on training since it can change
                #     kwe['iptc'] = kwe_iptc_map[k_uuid]
                results['kwe'][k_uuid] = kwe
    return results


def correct(arg) -> int:
    token = os.environ.get('KMAP_TOKEN')
    if not token:
        logger.error('Missing KMAP_TOKEN environment variable')
        return 1

    kwe_iptc_map = {}
    map_file_name = os.path.join(arg.result_dir, 'kwe_iptc_map.csv')
    with open(map_file_name, encoding='utf-8') as map_file:
        try:
            reader = csv.reader(map_file)
            for row in reader:
                key = row[0]
                kwe_iptc_map[key] = {'name': row[1], 'id': row[2]}
        except:
            logger.error("Unable to load CSV tag map file [%s].", map_file_name)
            return 1

    def callback(s: State, saved: Dict[str, Any]) -> int:
        topic_uuids = []

        def filter_tag(level: int, sub_dict: Dict[str, Any]):
            if level > 0:
                if 'refUuid' in sub_dict:
                    sub_dict['uuid'] = sub_dict.pop('refUuid', None)
                    sub_dict['type'] = 'topic_group'
                return True

            if 'type' in sub_dict and sub_dict['type'] == 'topic':
                topic_uuids.append(sub_dict['uuid'])
                return True
            else:
                return False

        _traverse_tags(0, saved, filter_tag)
        if not 'matched1' in saved['title']:
            kws = _get_keywords(
                topic_uuids, saved['uuid'], saved['title']['text'], saved['body']['text']
            )
            saved['title']['matches'] = _extract_intervals(
                kws['titleIntervals'], kwe_iptc_map
            )
            saved['body']['matches'] = _extract_intervals(
                kws['contentIntervals'], kwe_iptc_map
            )

        with open(s.file, 'w', encoding='utf8') as json_file:
            json.dump(saved, json_file, indent='  ', ensure_ascii=False)
        return 1

    state = load_range(arg.start_date, arg.end_date, arg.result_dir, callback)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    return 0
