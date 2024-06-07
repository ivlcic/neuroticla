import csv
import json
import logging
import os
from datetime import datetime

import pandas as pd
from typing import List, Dict, Any, Union

import requests
from transformers import AutoTokenizer, AutoModel

from .__embed import _embed, _filter_body
from .__utils import load_range, State, Params, filter_article, traverse_article_tags
from ...esdl import Elastika
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
                    'topic': {
                        'uuid': t_uuid,
                        'name': k['category']['metadata']['topicName']
                    },
                    # 'topic_name': k['category']['metadata']['topicName'],
                    'expr': k['category']['metadata']['regex']
                }
                # if k_uuid in kwe_iptc_map: #  better to map this on training since it can change
                #     kwe['iptc'] = kwe_iptc_map[k_uuid]
                results['kwe'][k_uuid] = kwe
    return results


def correct_old(arg) -> int:
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

    #if os.path.exists(arg.customers):
    #    with open(arg.customers) as f:
    #        customers = f.read().splitlines()
    #else:
    #    customers = arg.customers.split(',')
    customers = None

    params = Params(arg.start_date, arg.end_date, customers, arg.result_dir)

    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        if not os.path.exists(s.file):
            logger.warning("Unable to find article file %s", s.file)
            return 0
        if len(saved) == 1 and 'ver' in saved:
            logger.warning("Corrupted article file %s", s.file)
            os.remove(s.file)
            return 0
        if 'ver' in saved and (saved['ver'] == '1.1a' or saved['ver'] == '1.1c'):
            return 1
        logger.info("Correcting [%s]", s.file)
        el: Elastika = Elastika()
        el.limit(1)
        el.filter_uuid(saved['uuid'])
        articles = el.get(
            datetime.fromisoformat("2022-12-30").astimezone(),
            datetime.fromisoformat("2024-01-02").astimezone()
        )
        if len(articles) == 0:
            logger.warning("Unable to find article %s", s.file)
            os.remove(s.file)
            return 0
        article = articles[0]

        filter_article(article)
        saved['tags'] = article.data['tags']

        topic_uuids = []
        article_rates: Union[List, None] = saved.pop('rates', None)
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
                # if 'uuid' in sub_dict and sub_dict['uuid'] not in params.customersCtg:
                topic_uuids.append(sub_dict['uuid'])
                if sub_dict['uuid'] in article.data['topic_names']:
                    sub_dict['name'] = article.data['topic_names'][sub_dict['uuid']]

        params.tagCallback = filter_tag

        traverse_article_tags(params, 0, saved)
        if 'title' in saved and 'matches1' not in saved['title'] and len(topic_uuids):
            kws = _get_keywords(
                topic_uuids, saved['uuid'], saved['title']['text'], saved['body']['text']
            )
            saved['title']['matches'] = _extract_intervals(
                kws['titleIntervals'], kwe_iptc_map
            )
            saved['body']['matches'] = _extract_intervals(
                kws['contentIntervals'], kwe_iptc_map
            )
        if 'tags' not in saved:
            logger.warning("Article %s has no tags.", s.file)
            os.remove(s.file)
            return 0

        saved['ver'] = '1.1b'

        with open(s.file, 'w', encoding='utf8') as json_file:
            json.dump(saved, json_file, indent='  ', ensure_ascii=False)
        return 1

    state = load_range(params, callback)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    return 0


def correct(arg) -> int:
    token = os.environ.get('KMAP_TOKEN')
    if not token:
        logger.error('Missing KMAP_TOKEN environment variable')
        return 1

    #if os.path.exists(arg.customers):
    #    with open(arg.customers) as f:
    #        customers = f.read().splitlines()
    #else:
    #    customers = arg.customers.split(',')

    params = Params(arg.start_date, arg.end_date, None, arg.result_dir)

    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        if not os.path.exists(s.file):
            logger.warning("Unable to find article file %s", s.file)
            return 0
        if len(saved) == 1 and 'ver' in saved:
            logger.warning("Corrupted article file %s", s.file)
            os.remove(s.file)
            return 0
        if 'ver' in saved and saved['ver'] == '1.1c':
            return 1
        logger.info("Correcting [%s]", s.file)
        el: Elastika = Elastika()
        el.limit(1)
        el.filter_uuid(saved['uuid'])
        articles = el.get(
            datetime.fromisoformat("2022-12-30").astimezone(),
            datetime.fromisoformat("2024-01-02").astimezone()
        )
        if len(articles) == 0:
            logger.warning("Unable to find article %s", s.file)
            os.remove(s.file)
            return 0
        article = articles[0]

        filter_article(article)
        article_rates: Union[List, None] = article.data.pop('rates', None)
        rates = {}
        if article_rates is not None:
            for r in article_rates:
                rate_type = r['id']['rateType']
                rated = r['id']['beanUuid']
                if rate_type == 'CustomerTopic' or rate_type == 'CustomerTopicGroup':
                    rates[rated] = int(r['value'])

        # noinspection PyUnusedLocal
        for tag in saved['tags']:
            parent_uuid = tag.pop('parent', None)
            if parent_uuid is not None:
                tag['parent'] = {'uuid': parent_uuid}
                if parent_uuid in rates:
                    tag['parent']['sentiment'] = rates[parent_uuid]

        saved['ver'] = '1.1c'

        with open(s.file, 'w', encoding='utf8') as json_file:
            json.dump(saved, json_file, indent='  ', ensure_ascii=False)
        return 1

    state = load_range(params, callback)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    return 0


def _add_spans(sentiment: Dict[str, Any], type: str, tag_uuid: str, saved: Dict[str, Any]):
    for span in saved[type]['matches']['spans']:
        for kwe in span['kwe']:
            if kwe in saved[type]['matches']['kwe'] and \
                    saved[type]['matches']['kwe'][kwe]['topic']['uuid'] == tag_uuid:
                sent_span = {
                        "start": span['start'],
                        "end": span['end'],
                        "loc": type,
                        "value": saved[type]['text'][span['start']: span['end']]
                    }
                if sent_span not in sentiment['relevant_span_center_words']:
                    sentiment['relevant_span_center_words'].append(sent_span)


def sentiment(arg) -> int:
    customers = []
    params = Params(arg.start_date, arg.end_date, customers, arg.result_dir)

    map_file = os.path.join(arg.result_dir, 'customers_map.csv')
    customers_names = {}
    with open(map_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3:  # Ensure the row has at least two columns
                key = row[1]
                value = row[2]
                customers_names[key] = value
    missing = set()
    data = []
    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        if not os.path.exists(s.file):
            logger.warning("Unable to find article file %s", s.file)
            return 0
        if len(saved) == 1 and 'ver' in saved:
            logger.warning("Corrupted article file %s", s.file)
            os.remove(s.file)
            return 0
        path_parts = s.file.split(os.sep)
        relevant_parts = path_parts[path_parts.index('corpus') + 1:]

        body = ''
        filtered = False
        mt = saved['media']['type']['name']
        if mt == 'radio' or mt == 'tv':
            for line_idx, line in enumerate(saved['body']['text'].split('\n')):
                tmp = _filter_body(line_idx, line)
                if tmp:
                    body += '\n' if len(body) > 0 else ''
                    body += tmp
                if not tmp and line:
                    filtered = True
        if not body:
            body = saved['body']['text']

        new_data = {
            'uuid': saved['uuid'],
            'published': saved['published'],
            #'created': saved['created'],
            'country': saved['country']['name'],
            # 'media': saved['media'],
            # 'rubric': saved['rubric'],
            'url': saved['url'] if 'url' in saved else None,
            'language': saved['language'],
            # 'rel_path': os.path.join(*relevant_parts),
            'title': saved['title']['text'],
            'sentiments': [],
            'body': body,
            'ver': '1.1'
        }
        if not new_data['url']:
            new_data.pop('url', None)
        parents_visited = []
        for tag in saved['tags']:
            if 'parent' in tag and 'sentiment' in tag['parent'] and tag['parent']['uuid'] not in parents_visited:
                aspect = tag['parent']['uuid']
                if tag['parent']['uuid'] in customers_names:
                    aspect = customers_names[tag['parent']['uuid']]
                else:
                    missing.add(tag['parent']['uuid'])

                sentiment = {
                        'source': 'editors',
                        'value': tag['parent']['sentiment'],
                        'aspect': aspect,
                        'relevant_span_center_words': []
                    }
                new_data['sentiments'].append(sentiment)
                parents_visited.append(tag['parent']['uuid'])
                for other_tag in saved['tags']:
                    if other_tag['parent']['uuid'] == tag['parent']['uuid']:
                        _add_spans(sentiment, 'body', other_tag['uuid'], saved)
                        _add_spans(sentiment, 'title', other_tag['uuid'], saved)

        if not new_data['sentiments']:
            return 0
        data.append(new_data)
        return 1

    state = load_range(params, callback)

    save_dir = os.path.join(arg.result_dir, 'sentiment')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir,  f'sent_editors_{params.start.year}_{params.start.month:02d}.json')
    with open(file_name, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent='  ', ensure_ascii=False)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    logger.info("With missing %s", missing)
    return 0
