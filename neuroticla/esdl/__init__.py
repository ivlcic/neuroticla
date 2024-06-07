import uuid
from datetime import timedelta
from typing import Union, Iterable
from uuid import uuid3

import requests
from requests.auth import HTTPBasicAuth

from .article import *


class Elastika:

    def __init__(self, url: str = None, user: str = None, passwd: str = None):
        self._user: str = user if user else os.environ.get('CPTM_SUSER')
        self._passwd: str = passwd if passwd else os.environ.get('CPTM_SPASS')
        self._url: str = url if url else os.environ.get('CPTM_SURL')
        self._filters = []
        self._fields = [
            'uuid',
            'created',
            'published',
            'tags',
            'media',
            'mediaReach',
            'advertValue',
            'media.tags',
            'media.uuid',
            'country',
            'language',
            'translations'
        ]
        self._limit = 100
        self._offset = 0
        self._filter = {
            'uuids': [],
            'topics': [],
            'customers': [],
            'media_tags': [],
            'media': [],
            'prefix': [],
            'country': '',
            'language': '',
        }
        self._query_tpl: str = '''
        {
          "query": {
            "bool": {
              "filter": [
                {
                  "range": {
                    "created": {
                      "gte": "<date_start>",
                      "lt": "<date_end>"
                    }
                  }
                }
                <filters>
              ]
            }
          },
          "_source": [
            <fields>
          ],
          "from": <from>,
          "size": <size>,
          "sort": { "created": {"order": "asc"}}
        }
        '''

    def _inject_filters(self, query: str) -> str:
        tags = self._filter['topics'] + self._filter['customers']
        if tags:
            self._filters.append(
                '''
                {
                  "terms": {
                    "tags.uuid": ''' + json.dumps(tags) + '''
                  }
                }
                '''
            )
        if self._filter['uuids']:
            self._filters.append(
                '''
                {
                  "terms": {
                    "uuid": ''' + json.dumps(self._filter['uuids']) + '''
                  }
                }
                '''
            )
        if self._filter['country']:
            self._filters.append(
                '''
                {
                  "term": {
                    "country.name": "''' + self._filter['country'] + '''"
                  }
                }
                '''
            )
        if self._filter['language']:
            self._filters.append(
                '''
                {
                  "term": {
                    "language": "''' + self._filter['language'] + '''"
                  }
                }
                '''
            )
        if self._filter['media_tags']:
            self._filters.append(
                '''
                {
                  "terms": {
                    "media.tags.uuid": ''' + json.dumps(self._filter['media_tags']) + '''
                  }
                }
                '''
            )
        if self._filter['media']:
            self._filters.append(
                '''
                {
                  "terms": {
                    "media.uuid": ''' + json.dumps(self._filter['media']) + '''
                  }
                }
                '''
            )
        if self._filter['prefix']:
            prefixes = ['{"prefix": { "text": { "value": "' + x + '", "case_insensitive": true } }}'
                        for x in self._filter['prefix']]
            self._filters.append(
                '''
                {
                  "bool": {
                    "should": [
                      ''' + ','.join(prefixes) + '''
                    ],
                    "minimum_should_match": 1
                  }
                }
                '''
            )
        filters = ''
        if self._filters:
            filters = ',' + ','.join(self._filters)
        return query.replace('<filters>', filters)

    def filter_customer(self, customer_uuid: Union[None, str, Iterable[str]]) -> TArticles:
        if not customer_uuid:
            self._filter['customers'] = []
            return self
        if isinstance(customer_uuid, str):
            self._filter['customers'].append(str(uuid3(uuid.NAMESPACE_URL, 'CustomerTopicGroup.' + customer_uuid)))
            return self

        [self._filter['customers'].append(str(uuid3(uuid.NAMESPACE_URL, 'CustomerTopicGroup.' + x))) for x in customer_uuid]
        return self

    def filter_topic(self, topic_uuid: Union[None, str, Iterable[str]]) -> TArticles:
        if not topic_uuid:
            self._filter['topics'] = []
            return self
        if isinstance(topic_uuid, str):
            self._filter['topics'].append(topic_uuid)
            return self
        self._filter['topics'].extend(topic_uuid)
        return self

    def filter_uuid(self, a_uuid: Union[None, str, Iterable[str]]) -> TArticles:
        if not a_uuid:
            self._filter['uuids'] = []
            return self
        if isinstance(a_uuid, str):
            self._filter['uuids'].append(a_uuid)
            return self
        self._filter['uuids'].extend(a_uuid)
        return self

    def field(self, field: Union[None, str, Iterable[str]]) -> TArticles:
        if not field:
            self._fields = ['uuid']
            return self
        if isinstance(field, str):
            self._fields.append(field)
            return self
        self._fields.extend(field)
        return self

    def filter_country(self, code: str) -> TArticles:
        self._filter['country'] = code
        return self

    def filter_media_type(self, tag: str) -> TArticles:
        if not tag:
            self._filter['media_tags'] = []
            return self

        self._filter['media_tags'].append(tag)
        return self

    def filter_media(self, media_uuid: Union[None, str, Iterable[str]]) -> TArticles:
        if not media_uuid:
            self._filter['media'] = []
            return self
        if isinstance(media_uuid, str):
            self._filter['media'].append(media_uuid)
            return self
        self._filter['media'].extend(media_uuid)
        return self

    def filter_text(self, prefix: Union[None, str, Iterable[str]]) -> TArticles:
        if not prefix:
            self._filter['prefix'] = []
            return self
        if isinstance(prefix, str):
            self._filter['prefix'].append(prefix)
            return self
        self._filter['prefix'].extend(prefix)
        return self

    def limit(self, limit: int) -> TArticles:
        self._limit = limit
        return self

    def offset(self, offset: int) -> TArticles:
        self._offset = offset
        return self

    def get(self, start: datetime, end: datetime = None) -> List[Article]:
        if not end:
            end = start + timedelta(hours=24)
        query = self._query_tpl.replace('<from>', str(self._offset))
        query = query.replace('<size>', str(self._limit))
        query = query.replace('<date_start>', start.astimezone().isoformat())
        query = query.replace('<date_end>', end.astimezone().isoformat())
        query = self._inject_filters(query)
        fields = ''
        for idx, field in enumerate(self._fields):
            if idx > 0:
                fields += ','
            fields += '"' + field + '"'
        query = query.replace('<fields>', fields)
        logger.debug("Loading articles with Elasticsearch query: [%s]", query)
        result = []
        try:
            # make HTTP verb parameter case-insensitive by converting to lower()
            resp = requests.post(self._url,
                                 headers={'Content-Type': 'application/json'},
                                 auth=HTTPBasicAuth(self._user, self._passwd),
                                 data=query)
        except Exception as error:
            logger.error('Elasticsearch request [%s] error [%s]:', query, error)
            return result

        try:
            resp_text: Dict[str, Any] = json.loads(resp.text)
            for hit in resp_text['hits']['hits']:
                try:
                    result.append(Article(hit['_source']))
                except Exception as error:
                    logger.error('Elasticsearch parse error [%s]:', hit)
            logger.info("Loaded [%s] Elasticsearch articles.", len(result))
        except Exception as error:
            logger.error('Elasticsearch parse error [%s]:', resp.text)

        return result

    def gets(self, start: str, end: str = None) -> List[Article]:
        start_date = datetime.fromisoformat(start)
        end_date = None
        if end:
            end_date = datetime.fromisoformat(end)

        return self.get(start_date, end_date)
