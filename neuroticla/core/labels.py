import os
import numpy

from typing import Dict, List, Union

import pandas as pd


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


class BinaryLabeler(Labeler):

    def __init__(self, file_name: str = None, labels: List = None, replace_labels: Dict[str, str] = None):
        super().__init__(file_name, labels, replace_labels)
        self._source_labels = self._labels
        self._replace_labels = replace_labels
        if replace_labels is not None:
            for k in self._replace_labels.keys():
                self._labels = self._labels.remove(k)
        if len(self._labels) != 1:
            raise ValueError('BinaryLabeler should have single label!')
        self._label_to_id = {labels[0] + '-1': 1, labels[0] + '-0': 0}
        self._id_to_label = {1: labels[0] + '-1', 0: labels[0] + '-0'}

    def for_binary_eval(self) -> List[str]:
        result = []
        for lx in self._source_labels:
            result.append(lx + '-0')
            result.append(lx + '-1')

        return result


class MultiLabeler(Labeler):

    def __init__(self, file_name: str = None, labels: List = None, replace_labels: Dict[str, str] = None):
        super().__init__(file_name, labels, replace_labels)
        self._num_bits = len(self._source_labels)
        self._labels = []
        for i in range(0, 2 ** self._num_bits):
            bitlist = [k for k in range(i.bit_length()) if i & (1 << k)]
            label = ''
            for idx in bitlist:
                label += self._source_labels[idx]
            self._labels.append(label)
        self._label_to_id = {k: v for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}
        self._id_to_label = {v: k for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}

    def decode(self, idx: int) -> List[str]:
        labels = []
        for i in range(0, idx.bit_length()):
            mask = 1 << i
            test = idx & mask
            if test != 0:
                labels.append(self._source_labels[i])
        return labels

    def encode(self, labels: List[str]) -> int:
        target: str = ''
        for src_label in self._source_labels:
            if src_label in labels:
                target += src_label
        return self._label_to_id[target]

    def encode_columns(self, data: pd.DataFrame) -> List[int]:
        encoded = []
        for index, row in data.iterrows():
            label_list = [l for l in self._source_labels if row[l] == 1]
            encoded.append(self.encode(label_list))
        return encoded

    def for_binary_eval(self) -> List[str]:
        result = []
        for lx in self._source_labels:
            result.append(lx + '-0')
            result.append(lx + '-1')

        return result

    def binpowset(self, idx):
        if isinstance(idx, numpy.int64):
            idx = idx.item()
        labels = []
        for i in range(0, self._num_bits):
            mask = 1 << i
            test = idx & mask
            if test != 0:
                labels.append(1)
            else:
                labels.append(0)
        return labels
