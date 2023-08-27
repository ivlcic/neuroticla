import logging
import numpy as np

from unittest import TestCase
from argparse import ArgumentParser
from sklearn import metrics

from ..core.labels import MultiLabeler

logger = logging.getLogger('nf.test')


class MultilabelTest(TestCase):
    def init(self):
        labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
        self.assertEqual(['',
                          'eco',
                          'lab',
                          'ecolab',
                          'wel',
                          'ecowel',
                          'labwel',
                          'ecolabwel',
                          'sec',
                          'ecosec',
                          'labsec',
                          'ecolabsec',
                          'welsec',
                          'ecowelsec',
                          'labwelsec',
                          'ecolabwelsec',
                          'cul',
                          'ecocul',
                          'labcul',
                          'ecolabcul',
                          'welcul',
                          'ecowelcul',
                          'labwelcul',
                          'ecolabwelcul',
                          'seccul',
                          'ecoseccul',
                          'labseccul',
                          'ecolabseccul',
                          'welseccul',
                          'ecowelseccul',
                          'labwelseccul',
                          'ecolabwelseccul'], labeler.labels)

    def decode(self):
        labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
        labels = labeler.decode(18)
        self.assertEqual(['lab', 'cul'], labels)

        labels = labeler.decode(19)
        self.assertEqual(['eco', 'lab', 'cul'], labels)

        labels = labeler.decode(20)
        self.assertEqual(['wel', 'cul'], labels)

    def binpowset(self):
        labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
        labels = labeler.binpowset(18)
        self.assertEqual([0, 1, 0, 0, 1], labels)

        labels = labeler.binpowset(19)
        self.assertEqual([1, 1, 0, 0, 1], labels)

        labels = labeler.binpowset(20)
        self.assertEqual([0, 0, 1, 0, 1], labels)

    def encode(self):
        labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
        idx = labeler.encode(['eco', 'lab'])
        self.assertEqual(3, idx)

        idx = labeler.encode(['lab', 'eco'])
        self.assertEqual(3, idx)

        idx = labeler.encode(['cul', 'sec', 'lab'])
        self.assertEqual(26, idx)


def add_args(module_name: str, parser: ArgumentParser) -> None:
    pass


def unittest_init(arg):
    t = MultilabelTest()
    t.init()


def unittest_decode(arg):
    t = MultilabelTest()
    t.decode()


def unittest_binpowset(arg):
    t = MultilabelTest()
    t.binpowset()


def unittest_encode(arg):
    t = MultilabelTest()
    t.encode()


def unittest_metrics(arg):
    y_pred = [0, 1, 0, 0]
    y_true = [0, 1, 0, 1]
    f1_score = metrics.f1_score(y_true, y_pred, labels=['eco'])
    precision = metrics.precision_score(y_true, y_pred, labels=['eco'])
    recall = metrics.recall_score(y_true, y_pred, labels=['eco'])
    print(f'P:[{precision}], R:[{recall}], F1:{f1_score}')
    print(metrics.classification_report(y_true, y_pred))
    f1_score = metrics.f1_score(y_true, y_pred, labels=[0, 1], average='macro')
    precision = metrics.precision_score(y_true, y_pred, labels=['eco-0', 'eco-1'], average='macro')
    recall = metrics.recall_score(y_true, y_pred, labels=['eco-0', 'eco-1'], average='macro')
    print(f'P:[{precision}], R:[{recall}], F1:{f1_score}')

    y_pred = [0, 0, 0, 0, 0]
    y_true = [0, 0, 0, 1, 0]
    f1_score = metrics.f1_score(y_true, y_pred, labels=['eco'], zero_division=0)
    precision = metrics.precision_score(y_true, y_pred, labels=['eco'], zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, labels=['eco'], zero_division=0)
    print(f'P:[{precision}], R:[{recall}], F1:{f1_score}')
    print(metrics.classification_report(y_true, y_pred, zero_division=0))


def unittest_transpose(arg):
    # Create a 2D NumPy array
    arr = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1]
    ])
    print(arr)
    print(arr.transpose())


def unittest_sample(arg) -> int:
    pass