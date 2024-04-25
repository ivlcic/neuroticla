import logging
import pandas as pd
from argparse import ArgumentParser
from typing import List

from ..esdl import Elastika
from ..esdl.article import Article

logger = logging.getLogger('nf.slomcor')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    pass


def filter(a: Article) -> bool:
    return False

def main(arg) -> int:
    sources = [
        '1b64e062-3e83-4591-af86-a6e244c45ed5', '93045532-f197-4da9-9de0-be6795998d7e',
        'bc20546f-3a11-4061-90c2-2769468cd542', '07785797-5184-4963-9813-b6611846740a',
        'a6d81cfb-912a-426c-9bc1-845b52a46fe2', '4efb7ff6-78e6-4408-879e-59b9f092a8c9',
        '0360424e-26ac-4ef6-991b-d9756663b44f', '279ac7fb-4f94-4a04-8dd3-1546d47eedf7',
        '5a32cddf-b3bd-4698-88c1-1ef661fb43a6', 'b455f571-f0eb-4691-ad54-7d3dfb88f2bd',
        '76320d8f-543a-4b50-92c0-01453c885fd8', 'bf1d559b-8c27-4f32-8c2f-991d649527b2',
        '7ff71957-487f-456a-b0bb-db6029b8ba04', '6a996b7c-b8ea-4aa9-ab67-d8d558d1cba4',
        '685def26-7b7b-4089-b83c-990c06abd752', '9aedb9fd-2914-4d8e-9d63-1dbedd03cb65',
        '396b201f-2a65-44c8-857e-16c2bf79e55f', '1b498bcf-87b7-4888-b526-4807088c7738',
        '57537984-e949-45a1-b764-6fe4654aed54', '2b7bbd45-0ba5-4632-9716-43515f20bd6a',
        'a2d9c362-e25d-483a-a8f3-3b8d90fe05b9', 'c94af0ee-1f52-4f45-9eac-34ab40835a2f',
        '17d4af83-486f-4add-bd1e-b719ff6b9c2e', '29213ab4-199c-4b11-aa64-eaa37092adf6',
        'd53a5e20-a6dd-4ca5-b989-2b662b028f7b', 'e1ab11df-a6ad-4628-98e0-439376da009b',
        '2c055505-375b-47a3-a5f0-75c57d3cf9e2', 'ecd1daa4-1f1b-4259-9b03-105f0ba1ba00',
        '754da261-9aee-4a1a-b9d8-734cd409fabf'
    ]
    prefixes = [
        'begunec', 'begunc', 'begunk', 'beguns', 'migracij', 'migrant', 'imigra',
        'prebežni', 'pribežni', 'azil'
    ]

    date_start = '2010-01-01T00:00:00'
    date_end = '2017-01-01T00:00:00'

    esdl = Elastika()
    esdl.limit(10000)
    esdl.filter_country('SI')
    esdl.field('pageNumber')
    #esdl.filter_text(prefixes)
    #esdl.filter_media(sources)
    esdl.filter_customer('484a9f55-83db-45c3-b1a5-bc37a03e856c')

    articles1: List[Article] = esdl.gets(date_start, date_end)
    articles1 = [a for a in articles1 if len(a.title) > 0 and len(a.body) > 0]
    logger.info('1: %s', len(articles1))

    date_start = '2017-01-01T00:00:00'
    date_end = '2024-01-01T00:00:00'

    articles2: List[Article] = esdl.gets(date_start, date_end)
    articles2 = [a for a in articles2 if len(a.title) > 0 and len(a.body) > 0]
    logger.info('1: %s', len(articles2))

    articles1.extend(articles2)

    index = {}
    invalid_titles = [
        'pomembne številke', 'mali oglasi', 'oglasi', 'oglas', 'ne', 'bob dneva', 'tv spored', 'razstave', 'znanje',
        'prireditve', 'prosta delovna mesta (s pogoji za zasedbo)', 'delova borza dela', 'glasba'
        'borza dela in znanja', 'borza dela', 'radio', 'tv', 'predavanja', 'drugo', 'novi doktorji znanosti'
        'sporedi', 'tv spored nedelja', 'tv sporedi', 'osmrtnice', 'tv najave', 'vroča delovna mesta', 'kam po pomoč',
        'drugi dogodki', 'ujeti v preteklost', 'obvestila', 'misel dneva', 'prosta delovna mesta', 'plinarna maribor',
        'popravek', 'drugo', 'izjava dneva', 'razpis'
    ]
    for article in articles1:
        title_trimmed = article.title.strip()
        title = title_trimmed.lower()
        #if title == 'novi doktorji znanosti':
        if title in invalid_titles or 'naslovna stran' in title \
                or 'prva stran' in title or title_trimmed.endswith("NE") \
                or 'tv spored' in title \
                or 'prosta delovna mesta' in title \
                or 'novi doktorji znanosti' in title:
            continue
        logger.info('Some : %s -> %s', article.title, title)
        idx_name = article.published.strftime('%Y-%m-%d') + '--' + article.media + '--' + title
        if idx_name not in index:
            index[idx_name] = []
        index[idx_name].append(article)

    articles = []
    for k, v in index.items():
        if len(v) == 1:
            articles.append(v[0])
        if len(v) == 2:
            non_ijs_1 = [x for x in v[0].topics if x['uuid'] != 'f6de088c-9981-48f1-a281-f9406f57d28d']
            non_ijs_2 = [x for x in v[1].topics if x['uuid'] != 'f6de088c-9981-48f1-a281-f9406f57d28d']
            if len(non_ijs_1) > len(non_ijs_2):
                logger.info('Duplicate 0: %s -> %s', k, v[0])
                articles.append(v[0])
            else:
                logger.info('Duplicate 1: %s -> %s', k, v[1])
                articles.append(v[1])

    lst = []
    for article in articles:
        lst.append({
            'uuid': article.uuid,
            'created': article.created,
            'country': article.country,
            'published': article.published,
            'media': article.media,
            'media_type': article.mediaType['name'],
            'title': article.title,
            'body': article.body
        })
    df = pd.DataFrame(lst)
    df.to_csv('/home/nikola/social_services.cvs', encoding='utf-8')

    lst = []
    for article in articles:
        #non_ijs = [x for x in article.topics if x['uuid'] != 'f6de088c-9981-48f1-a281-f9406f57d28d']
        #if len(non_ijs) == 0:
        #    article.title = 'REMOVE: ' + article.title
        lst.append({
            'uuid': article.uuid,
            'page': article.data['pageNumber'] if 'pageNumber' in article.data else 0,
            'created': article.created,
            'country': article.country,
            'published': article.published,
            'media': article.media,
            'media_type': article.mediaType['name'],
            'title': article.title
        })
    df = pd.DataFrame(lst)
    df.to_csv('/home/nikola/social_services_idx.cvs', encoding='utf-8')
    return 0