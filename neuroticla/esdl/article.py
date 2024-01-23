import json
import logging
import os
from datetime import datetime
from typing import Any, List, TypeVar, Dict

from mergedeep import merge

logger = logging.getLogger('esdl.article')

TArticle = TypeVar("TArticle", bound="Article")
TArticles = TypeVar("TArticles", bound="Articles")


class Article:

    def __init(self):
        if 'uuid' in self.data:
            self.uuid = self.data['uuid']
        if 'url' in self.data:
            self.url = self.data['url']
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
                        self.mediaType = tag

        if 'tags' in self.data:
            for tag in self.data['tags']:
                if 'org.dropchop.jop.beans.tags.CustomerTopicGroup' == tag['class']:
                    self.customers.append(tag)
                if 'org.dropchop.jop.beans.tags.CustomerTopic' == tag['class']:
                    self.topics.append(tag)

    def __init__(self, json_object: Dict[str, Any]):
        self.data: Dict[str, Any] = json_object
        self.uuid: str = ''
        self.url: str = ''
        self.language: str = ''
        self.title: str = ''
        self.body: str = ''
        self.media: str = ''
        self.mediaReach: int = 0
        self.advertValue: float = 0.0
        self.mediaType: Dict[str, Any] = {}
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
        with open(os.path.join(data_path, self.uuid + '.json'), 'w', encoding='utf8') as json_file:
            json.dump(self.data, json_file, indent='  ', ensure_ascii=False)

    def from_cache(self, data_path) -> bool:
        if not os.path.exists(data_path):
            return False
        fname = os.path.join(data_path, self.uuid + '.json')
        if not os.path.exists(fname):
            return False
        with open(fname, encoding='utf8') as file:
            json_object = json.loads(file.read())
            merge(json_object, self.data)
            self.data = json_object
            self.__init()
            return True
