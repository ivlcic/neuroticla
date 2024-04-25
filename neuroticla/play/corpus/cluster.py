import json
import os
import logging

from datetime import datetime, timedelta
from typing import Dict, List

from .__utils import load_range, State, Params
from ...esdl.article import Article
from ..utils import cluster_louvain, cluster_print_xlsx, cluster_prep_json, cluster_create_wb, cluster_print_sheet

logger = logging.getLogger('play.corpus.cluster')


def dump(arg) -> int:
    collected = []
    params = Params(arg.start_date, arg.end_date, [], arg.result_dir)
    def callback(s: State, saved_article: Dict, a: Article) -> int:
        if s.index == 0:
            s.log = {'scanned': 0, 'kept': 0}
        s.log['scanned'] += 1
        if arg.country is not None:
            if saved_article['country']['name'] != arg.country:
                return 0
        data = {
            'ebd': saved_article['embed_oai']
        }
        article = Article(data)
        article.country = {'name': saved_article['country']['name']}
        article.language = saved_article['language']
        article.title = saved_article['title']['text']
        article.body = saved_article['body']['text']
        article.uuid = saved_article['uuid']
        article.data['relPath'] = s.file[len(arg.result_dir) + 1:]
        article.published = datetime.fromisoformat(saved_article['published'])
        article.created = datetime.fromisoformat(saved_article['created'])
        article.media = saved_article['media']['name']
        article.rubric = saved_article['rubric']['name']
        if 'mediaReach' in saved_article['media']:
            article.mediaReach = saved_article['media']['mediaReach']
        else:
            article.mediaReach = 0
        article.mediaType = saved_article['media']['type']
        if 'url' in saved_article:
            article.url = saved_article['url']
        # article.body = saved_article['body']['text']
        collected.append(article)
        s.log['kept'] += 1
        return 1

    state = load_range(params, callback)

    logger.info(
        "Collected [%s:%s] files [%s::%s] ", len(collected), state.total, state.start, state.end
    )
    for i in range(3):
        threshold = 0.95 + (0.01 * i)
        clusters = cluster_louvain(collected, 'ebd', threshold)
        logger.info(
            "Computed [%s] clusters [%s::%s] ", len(clusters), state.start, state.end
        )
        cluster_print_xlsx(
            clusters, os.path.join(
                arg.result_dir, f'clusters-{arg.start_date}_{arg.end_date}_{(int(threshold * 100))}.xlsx'
            )
        )
        data = cluster_prep_json(
            clusters,
            arg.country,
            state.start,
            state.end
        )
        file_name = os.path.join(
            arg.result_dir, f'clusters-{arg.start_date}_{arg.end_date}_{(int(threshold * 100))}.json'
        )
        with open(file_name, 'wt') as fp:
            json.dump(data, fp, indent=2)
        logger.info(
            "Done [%s] clusters [%s::%s] ", len(clusters), state.start, state.end
        )
    return 0


def dump2(arg) -> int:
    collected: Dict[str, List[Article]] = {}
    num_days = 5

    params = Params(arg.start_date, arg.end_date, [], arg.result_dir)
    def callback(s: State, saved_article: Dict, a: Article) -> int:
        if s.index == 0:
            s.log = {'scanned': 0, 'kept': 0}
        s.log['scanned'] += 1
        if arg.country is not None:
            if saved_article['country']['name'] != arg.country:
                return 0
        data = {
            'ebd': saved_article['embed_oai']
        }
        article = Article(data)
        article.country = {'name': saved_article['country']['name']}
        article.language = saved_article['language']
        article.title = saved_article['title']['text']
        article.body = saved_article['body']['text']
        article.uuid = saved_article['uuid']
        article.data['relPath'] = s.file[len(arg.result_dir) + 1:]
        article.published = datetime.fromisoformat(saved_article['published'])
        article.created = datetime.fromisoformat(saved_article['created'])
        article.media = saved_article['media']['name']
        article.rubric = saved_article['rubric']['name']
        if 'mediaReach' in saved_article['media']:
            article.mediaReach = saved_article['media']['mediaReach']
        else:
            article.mediaReach = 0
        article.mediaType = saved_article['media']['type']
        if 'url' in saved_article:
            article.url = saved_article['url']

        delta = article.created - params.start
        bucket_start_date = params.start + timedelta(days=(delta.days // num_days) * num_days)
        bucket_key = bucket_start_date.strftime('%Y-%m-%d')
        if bucket_key not in collected:
            collected[bucket_key] = []
        collected[bucket_key].append(article)
        s.log['kept'] += 1
        return 1

    state = load_range(params, callback)

    save_dir = os.path.join(arg.result_dir, 'cluster')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger.info(
        "Collected [%s:%s] files [%s::%s] ", len(collected), state.total, state.start, state.end
    )
    for i in range(4):
        threshold = 0.94 + (0.01 * i)
        clusters = {}
        wb = cluster_create_wb()

        for key, articles in collected.items():
            max_datetime = max(articles, key=lambda obj: obj.created).created
            min_datetime = min(articles, key=lambda obj: obj.created).created
            daily_clusters = cluster_louvain(articles, 'ebd', threshold)
            logger.info(
                "Computed [%s] %s days clusters [%s from %s::%s] ",
                len(daily_clusters), num_days, key, state.start, state.end
            )
            cluster_print_sheet(wb, key, daily_clusters)
            data = cluster_prep_json(
                daily_clusters,
                arg.country,
                min_datetime,
                max_datetime
            )
            file_name = os.path.join(
                save_dir, f'clusters-{key}_{(int(threshold * 100))}.json'
            )
            #with open(file_name, 'wt') as fp:
            #    json.dump(data, fp, indent=2)
            clusters[key] = data
            logger.info(
                "Done [%s] %s days clusters [%s from %s::%s] ",
                len(daily_clusters), num_days, key, state.start, state.end
            )
        file_name = os.path.join(
            save_dir, f'{num_days}days_clusters_{(int(threshold * 100))}.json'
        )
        with open(file_name, 'wt') as fp:
            json.dump(clusters, fp, indent=2)

        file_name = os.path.join(
            save_dir, f'{num_days}days_clusters_{(int(threshold * 100))}.xlsx'
        )
        wb.save(file_name)

    return 0
