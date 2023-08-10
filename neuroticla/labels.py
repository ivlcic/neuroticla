import os

from typing import Dict, List, Union


class Labeler:

    def __init__(self, file_name: str = None, labels: List[str] = None, replace_labels: Dict[str, str] = None):
        if replace_labels is None:
            replace_labels = {}
        if labels is None:
            labels = []

        if file_name is not None and os.path.exists(file_name):
            with open(file_name, "r", encoding='utf-8') as fp:
                self._labels = fp.read().splitlines()
        else:
            self._labels = labels
        if not self._labels:
            raise ValueError('Either valid file_name or labels list must be present')
        self._source_labels = self._labels
        self._replace_labels = replace_labels
        self._label_to_id = {k: v for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}
        self._id_to_label = {v: k for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}

    def label2id(self, label: str) -> int:
        if not label:
            return -1
        if label in self._label_to_id:
            return self._label_to_id[label]
        return -1

    def labels2ids(self):
        return self._label_to_id

    def id2label(self, _id: int, default: Union[str, None] = None) -> str:
        if _id in self._id_to_label:
            return self._id_to_label[_id]
        return default

    def ids2labels(self):
        return self._id_to_label

    def kept_labels(self):
        return self._label_to_id.keys()

    def labels(self):
        return self._label_to_id.keys()

    def source_labels(self):
        return self._source_labels

    def mun_labels(self):
        return len(self._label_to_id.keys())

    def filter_replace(self, text: str):
        for k, v in self._replace_labels.items():
            text = text.replace(k, v)
        return text