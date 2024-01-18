import os
import logging

from datetime import datetime
from typing import Dict

from .__utils import load_range, State
from ...esdl.article import Article
from ..utils import cluster_louvain, cluster_print_xlsx, cluster_print_json

logger = logging.getLogger('play.corpus.cluster')


def dump(arg) -> int:
    collected = []

    def callback(s: State, saved_article: Dict) -> int:
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

    state = load_range(arg.start_date, arg.end_date, arg.result_dir, callback)

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
        cluster_print_json(
            clusters,
            arg.country,
            state.start,
            state.end,
            os.path.join(
                arg.result_dir, f'clusters-{arg.start_date}_{arg.end_date}_{(int(threshold * 100))}.json'
            )
        )
        logger.info(
            "Done [%s] clusters [%s::%s] ", len(clusters), state.start, state.end
        )
    return 0
