import csv
import json
import logging
import os
from typing import List, Dict, Any, Union

import requests
from transformers import AutoTokenizer, AutoModel

from .__embed import _embed
from .__utils import load_range, State, Params, filter_article, traverse_article_tags
from ...esdl.article import Article

logger = logging.getLogger('play.corpus.dump')


def dump(arg) -> int:
    model_name = 'intfloat/multilingual-e5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=os.path.join(arg.tmp_dir, model_name)
    )

    if os.path.exists(arg.customers):
        with open(arg.customers) as f:
            customers = f.read().splitlines()
    else:
        customers = arg.customers.split(',')

    params = Params(arg.start_date, arg.end_date, customers, arg.result_dir)
    params.set_model(model)
    params.set_tokenizer(tokenizer)
    params.skipEmbedding = True

    # noinspection PyUnusedLocal
    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        filter_article(article)
        traverse_article_tags(params, 0, article.data)
        if not os.path.exists(s.file):
            # write article for the first time
            _embed(params, article)
            with open(s.file, 'w', encoding='utf8') as json_file:
                json.dump(article.data, json_file, indent='  ', ensure_ascii=False)
        return 1

    state = load_range(params, callback)

    logger.info(
        "Dumping [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    return 0


def _get_keywords(topic_uuids: List[str], uuid, title, content) -> Dict[str, Any]:
    req_data = {
        "provider": "admin",
        "articleUuid": uuid,
        "title": title,
        "content": content,
        "includeTopics": topic_uuids,
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


# noinspection PyUnusedLocal
def _extract_intervals(matched: List[Dict[str, Any]], kwe_iptc_map: Dict[str, Dict[str, str]]):
    results = {
        'kwe': {},
        'spans': []
    }
    for m in matched:
        result = {'start': m['start'], 'end': m['end'], 'kwe': []}
        results['spans'].append(result)
        for k in m['matches']:
            k_uuid = k['category']['id']
            t_uuid = k['category']['metadata']['topicId']
            if k_uuid not in result['kwe']:
                result['kwe'].append(k_uuid)
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
        # noinspection PyBroadException
        try:
            reader = csv.reader(map_file)
            for row in reader:
                key = row[0]
                kwe_iptc_map[key] = {'name': row[1], 'id': row[2]}
        except Exception:
            logger.error("Unable to load CSV tag map file [%s].", map_file_name)
            return 1

    if os.path.exists(arg.customers):
        with open(arg.customers) as f:
            customers = f.read().splitlines()
    else:
        customers = arg.customers.split(',')

    params = Params(arg.start_date, arg.end_date, customers, arg.result_dir)

    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        if 'ver' in saved and saved['ver'] == '1.1d':
            return 1
        filter_article(article)
        topic_uuids = []
        article_rates: Union[List, None] = article.data.pop('rates', None)
        rates = {}
        if article_rates is not None:
            for r in article_rates:
                rate_type = r['id']['rateType']
                rated = r['id']['beanUuid']
                if rate_type == 'CustomerTopic' or rate_type == 'CustomerTopicGroup':
                    rates[rated] = int(r['value'])

        # noinspection PyUnusedLocal
        def filter_tag(level: int, sub_dict: Dict[str, Any]):
            if 'uuid' in sub_dict and sub_dict['uuid'] in rates:
                sub_dict['sentiment'] = rates[sub_dict['uuid']]
            if 'refUuid' in sub_dict and sub_dict['refUuid'] in rates:
                sub_dict['sentiment'] = rates[sub_dict['refUuid']]
            if 'type' in sub_dict and sub_dict['type'] == 'topic':
                if 'uuid' in sub_dict and sub_dict['uuid'] not in params.customersCtg:
                    topic_uuids.append(sub_dict['uuid'])

        params.tagCallback = filter_tag
        traverse_article_tags(params, 0, saved)
        if 'title' in saved and 'matches1' not in saved['title']:
            kws = _get_keywords(
                topic_uuids, saved['uuid'], saved['title']['text'], saved['body']['text']
            )
            saved['title']['matches'] = _extract_intervals(
                kws['titleIntervals'], kwe_iptc_map
            )
            saved['body']['matches'] = _extract_intervals(
                kws['contentIntervals'], kwe_iptc_map
            )
        saved['ver'] = '1.1d'

        with open(s.file, 'w', encoding='utf8') as json_file:
            json.dump(saved, json_file, indent='  ', ensure_ascii=False)
        return 1

    state = load_range(params, callback)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    return 0
