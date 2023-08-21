import os
import requests
import json
import logging
import uuid

from mergedeep import merge
from uuid import uuid3
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from typing import Any, List, TypeVar, Dict, Union, Iterable

logger = logging.getLogger('esdl.articles')

TArticle = TypeVar("TArticle", bound="Article")
TArticles = TypeVar("TArticles", bound="Articles")


class Article:

    def __init(self):
        if 'uuid' in self.data:
            self.uuid = self.data['uuid']
        if 'created' in self.data:
            self.created = datetime.fromisoformat(self.data['created'].replace('Z', '+00:00'))
        if 'published' in self.data:
            self.published = datetime.fromisoformat(self.data['published'].replace('Z', '+00:00'))
        if 'language' in self.data:
            self.language = self.data['language']
            if 'translations' in self.data and self.language in self.data['translations']:
                self.title = self.data['translations'][self.language]['title']
                try:
                    self.body = self.data['translations'][self.language]['body']
                except Exception as e:
                    brek = 1

        if 'country' in self.data:
            self.country = self.data['country']['name']

        if 'mediaReach' in self.data:
            self.mediaReach = self.data['mediaReach']
        if 'advertValue' in self.data:
            self.advertValue = self.data['advertValue']

        if 'media' in self.data:
            self.media = self.data['media']['name']
            if 'tags' in self.data['media']:
                for tag in self.data['media']['tags']:
                    if 'org.dropchop.jop.beans.tags.MediaType' == tag['class']:
                        self.media_type = tag

        if 'tags' in self.data:
            for tag in self.data['tags']:
                if 'org.dropchop.jop.beans.tags.CustomerTopicGroup' == tag['class']:
                    self.customers.append(tag)
                if 'org.dropchop.jop.beans.tags.CustomerTopic' == tag['class']:
                    self.topics.append(tag)

    def __init__(self, json_object: Dict[str, Any]):
        self.data: Dict[str, Any] = json_object
        self.uuid: str = ''
        self.language: str = ''
        self.title: str = ''
        self.body: str = ''
        self.media: str = ''
        self.mediaReach: int = 0
        self.advertValue: float = 0.0
        self.media_type: Dict[str, Any] = {}
        self.customers: List = []
        self.topics: List = []
        self.country: str = ''
        self.created: datetime = datetime(1999, 1, 1, 0, 0, 0, 0)
        self.published: datetime = datetime(1999, 1, 1, 0, 0, 0, 0)
        self.__init()

    def __str__(self) -> str:
        return '[' + self.uuid + '][' + self.created.astimezone().isoformat(timespec='seconds') \
            + '][' + self.country + '][' + self.media + '][' + self.title + ']'

    def __eq__(self, other):
        if isinstance(other, Article):
            return self.uuid == other.uuid
        return NotImplemented

    def __hash__(self):
        if self.uuid:
            return self.uuid.__hash__()
        return NotImplemented

    def to_cache(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        with open(os.path.join('data', self.uuid + '.json'), 'w', encoding='utf8') as json_file:
            json.dump(self.data, json_file, indent='  ', ensure_ascii=False)

    def from_cache(self, data_path) -> bool:
        if not os.path.exists(data_path):
            return False
        fname = os.path.join('data', self.uuid + '.json')
        if not os.path.exists(fname):
            return False
        with open(fname, encoding='utf8') as file:
            json_object = json.loads(file.read())
            merge(json_object, self.data)
            self.data = json_object
            self.__init()
            return True


class Esdl:

    def __init__(self, url: str = None, user: str = None, passwd: str = None):
        self._user: str = user if user else os.environ['CPTM_SUSER']
        self._passwd: str = passwd if passwd else os.environ['CPTM_SPASS']
        self._url: str = url if url else os.environ['CPTM_SURL']
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
          "size": <size>
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

    def filter_customer(self, customer_uuid: str) -> TArticles:
        if not customer_uuid:
            self._filter['customers'] = []
            return self
        if isinstance(customer_uuid, str):
            self._filter['customers'].append(str(uuid3(uuid.NAMESPACE_URL, 'CustomerTopicGroup.' + customer_uuid)))
            return self

        [self._filter['topics'].append(str(uuid3(uuid.NAMESPACE_URL, 'CustomerTopicGroup.' + x))) for x in customer_uuid]
        return self

    def filter_topic(self, topic_uuid: Union[str, Iterable[str]]) -> TArticles:
        if not topic_uuid:
            self._filter['topics'] = []
            return self
        if isinstance(topic_uuid, str):
            self._filter['topics'].append(topic_uuid)
            return self
        self._filter['topics'].extend(topic_uuid)
        return self

    def filter_uuid(self, a_uuid: Union[str, Iterable[str]]) -> TArticles:
        if not a_uuid:
            self._filter['uuids'] = []
            return self
        if isinstance(a_uuid, str):
            self._filter['uuids'].append(a_uuid)
            return self
        self._filter['uuids'].extend(a_uuid)
        return self

    def field(self, field: Union[str, Iterable[str]]) -> TArticles:
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

    def filter_media(self, media_uuid: Union[str, Iterable[str]]) -> TArticles:
        if not media_uuid:
            self._filter['media'] = []
            return self
        if isinstance(media_uuid, str):
            self._filter['media'].append(media_uuid)
            return self
        self._filter['media'].extend(media_uuid)
        return self

    def filter_text(self, prefix: Union[str, Iterable[str]]) -> TArticles:
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
                result.append(Article(hit['_source']))
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
