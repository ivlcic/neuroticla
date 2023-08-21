import logging
import uuid

from typing import List
from ..esdl.article import Article
from .api import call_textonic

logger = logging.getLogger('ttnx.embed')


def __call_ttxn_embed(articles: List[Article], embed_field_name: str,
                      average_t: str = None, weight_t: str = None):
    attrs = []
    if average_t is not None:
        attrs.append({"avg": average_t})
        if weight_t is not None:
            attrs.append({"weight": weight_t})
    request = {
        'requestId': str(uuid.uuid4()),
        'process': {
            'analysis': {
                'steps': [
                    {
                        'step': 'doc_embed',
                        'attributes': [
                            {'named_sentence_filters': 'kl_transcript'}
                        ]
                    }
                ]
            }
        },
        'documents': []
    }
    if attrs:
        request['process']['analysis']['steps'][0]['attributes'] = attrs

    for a in articles:
        document = {
            'id': a.uuid,
            'title': a.title,
            'lang': a.language,
            'sections': [
                {
                    'outline': 'headline',
                    'data': a.title
                },
                {
                    'outline': 'body',
                    'data': a.body
                }
            ]
        }
        request['documents'].append(document)
    logger.debug('Loading [%s] articles Textonic embedding ...', len(articles))
    resp_obj = call_textonic('/api/public/ml/process', request)

    for res_item, a in zip(resp_obj['data'], articles):
        for res in res_item['result']:
            if 'c' in res and 'v' in res and 'doc_embed' in res['c']:
                a.data[embed_field_name] = res['v']

    logger.info('Loaded [%s] articles Textonic embeddings.', len(articles))


def ttnx_embed(articles: List[Article], embed_field_name: str, cache: bool = True,
               average_t: str = None, weight_t: str = None):
    embed = []
    for a in articles:
        if cache and a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding
                logger.debug('Loaded %s article Textonic sentence BERT embedding from cache.', a)
                continue
        embed.append(a)
    if not embed:
        return
    __call_ttxn_embed(embed, embed_field_name, average_t, weight_t)
    if cache:
        for a in embed:
            a.to_cache('data')  # cache article to file
