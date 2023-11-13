import logging
import json
import numpy as np

from unittest import TestCase
from argparse import ArgumentParser
from sklearn import metrics

from ..core.labels import MultiLabeler
from ..core.eval import ClassificationMetrics
from ..core.split import DataSplit

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
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    labels = ['eco']
    f1_score = metrics.f1_score(y_true, y_pred, labels=labels)
    precision = metrics.precision_score(y_true, y_pred, labels=labels)
    recall = metrics.recall_score(y_true, y_pred, labels=labels)

    print(metrics.classification_report(y_true, y_pred))
    print(f'P:[{precision}], R:[{recall}], F1:{f1_score}')
    print('==============================================================================================')

    ny_pred = np.array([y_pred]).transpose().tolist()  # convert to single column matrix - vector
    ny_true = np.array([y_true]).transpose().tolist()
    m = ClassificationMetrics()
    report = m.compute(ny_true, ny_pred, labels=labels)
    report_str = json.dumps(report, indent=2)
    print(f'Result: {report_str}')
    print('==============================================================================================')

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


def unittest_c_report(arg) -> int:
    y_true = [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 1]
    ]
    y_pred = [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1]
    ]
    print(metrics.classification_report(y_true, y_pred, digits=3, target_names=['eco', 'lab', 'wel', 'sec']))
    print(metrics.precision_score(y_true, y_pred, labels=['eco', 'lab', 'wel', 'sec']))
    print(metrics.f1_score(y_true, y_pred, average='macro'))
    print(metrics.f1_score(y_true, y_pred, average='micro'))
    print(metrics.f1_score(y_true, y_pred, average='weighted'))
    return 0


def unittest_pandas(arg) -> int:
    import pandas as pd

    # Create a sample DataFrame with an additional numerical column ('inventory')
    data = {
        'country': ['USA', 'USA', 'USA', 'USA', 'India', 'India', 'India'],
        'media': ['TV', 'Online', 'TV', 'TV', 'Online', 'TV', 'Newspaper'],
        'sales': [100, 150, 200, 50, 75, 50, 25],
        'eco': [1, 0, 1, 1, 0, 1, 0],
        'lab': [1, 0, 1, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)

    # Perform multiple aggregation operations in one go
    grouped_df = df.groupby(['country', 'media']).agg(
        count=pd.NamedAgg(column='sales', aggfunc='count'),
        eco=pd.NamedAgg(column='eco', aggfunc='sum'),
        lab=pd.NamedAgg(column='lab', aggfunc='sum')
    ).reset_index()
    print(grouped_df)
    return 0


def unittest_chi2(arg) -> int:
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency([[1161, 718], [870, 1272], [950, 1183], [1908, 1797]])
    print(f'chi2 [{chi2}], p[{p:.6f}]')
    chi2, p, _, _ = chi2_contingency([[730, 718], [870, 883], [950, 953], [1908, 1808]])
    print(f'chi2 [{chi2}], p[{p:.6f}]')
    return 0


def unittest_search(arg) -> int:
    df = DataSplit.read_csv('/home/nikola/projects/neuroticla/data/nf/split/aussda/aussda_manual.csv')
    row = df[df['id'] == 1046038]
    selected_columns = df.loc[df['id'] == 1046038, ['id', 'title', 'lead', 'body', 'eco', 'lab', 'wel', 'sec']]
    print('---------------------------------------------')
    print(selected_columns.to_dict())
    print('---------------------------------------------')
    return 0
