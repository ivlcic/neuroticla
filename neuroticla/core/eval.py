from typing import Union, List, Any, Dict

import numpy as np
from sklearn import metrics


class ClassificationMetrics:

    @classmethod
    def flatten_1hot(cls, indicators: Union[List, np.array], binary: bool = True) -> List:
        result = []
        for indicator in indicators:
            for i, v in enumerate(indicator):
                result.append((i * (2 if binary else 1)) + (0 if v == 0 else 1))
        return result

    @classmethod
    def score(cls, references: List[Any], predictions: List[Any], labels: List[str], average: str = 'binary'):
        f1 = metrics.f1_score(references, predictions, zero_division=0, labels=labels, average=average)
        precision = metrics.precision_score(references, predictions, zero_division=0, labels=labels, average=average)
        recall = metrics.recall_score(references, predictions, zero_division=0, labels=labels, average=average)
        result = {
            'f1': f1,
            'p': precision,
            'r': recall,
            's': len(references)
        }
        if average == 'binary':
            result['a'] = metrics.accuracy_score(references, predictions)
        return result

    @classmethod
    def for_binary_eval(cls, labels: List[str]) -> List[str]:
        return [lx + suffix for lx in labels for suffix in ['-0', '-1']]

    def compute(self, references: List[Any], predictions: List[Any], labels: List[str], output_dict: bool = True):
        result: Dict[str, Any] = {
            'avg': {
                'micro': {},
                'macro': {},
                'weighted': {}
            },
            'accuracy': 0,
            'hamming': 0,
            'labels': {}
        }
        bin_labels = ClassificationMetrics.for_binary_eval(labels)
        if len(predictions) > 0 and (isinstance(predictions[0], List) or isinstance(predictions[0], np.ndarray)):
            # we assume label indicator array / sparse matrix and two times more labels than sample dimensions
            y_pred = ClassificationMetrics.flatten_1hot(predictions)
            y_true = ClassificationMetrics.flatten_1hot(references)
        else:
            y_pred = predictions
            y_true = references

        cr: Dict[str, Any] = metrics.classification_report(
            y_true, y_pred, digits=3, zero_division=0, output_dict=True, target_names=bin_labels
        )
        l_indices = []
        for i, label in enumerate(bin_labels):
            result['labels'][label] = {
                'f1': cr[label]['f1-score'], 'p': cr[label]['precision'],
                'r': cr[label]['recall'], 's': cr[label]['support']
            }
            l_indices.append(i)

        result['avg']['macro'] = {
            'f1': cr['macro avg']['f1-score'], 'p': cr['macro avg']['precision'],
            'r': cr['macro avg']['recall'], 's': cr['macro avg']['support']
        }
        result['avg']['weighted'] = {
            'f1': cr['weighted avg']['f1-score'], 'p': cr['weighted avg']['precision'],
            'r': cr['weighted avg']['recall'], 's': cr['weighted avg']['support']
        }
        result['avg']['micro'] = ClassificationMetrics.score(y_true, y_pred, l_indices, 'micro')
        if 'accuracy' not in cr:
            result['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        else:
            result['accuracy'] = cr['accuracy']
        result['hamming'] = metrics.hamming_loss(references, predictions)

        return result


class MultilabelMetrics:

    @classmethod
    def _add_prfs(cls, d, p, r, f1, s):
        # just adds values p, r, f1, s to dictionary with nice names
        d['f1'] = f1
        d['p'] = p
        d['r'] = r
        d['s'] = s

    @classmethod
    def _add_tnfpfntp(cls, d, tn, fp, fn, tp):
        # just adds values tp tn fp fn to dictionary
        d['tn'] = tn
        d['fp'] = fp
        d['fn'] = fn
        d['tp'] = tp

    def compute(self, references: Union[List[Any], np.ndarray], predictions: Union[List[Any], np.ndarray],
                labels: List[str], output_dict: bool = True):
        y_true: np.ndarray = np.array(references)
        y_pred: np.ndarray = np.array(predictions)
        yt_true = y_true.transpose()
        yt_pred = y_pred.transpose()
        report = {'labels': {}, 'avg': {'micro': {}, 'macro': {}, 'weighted': {}}}

        sum_pos_pred = 0
        for lx, lb in enumerate(labels):
            report['labels'][lb] = {}
            s = yt_true[lx].sum()
            p, r, f1, _ = metrics.precision_recall_fscore_support(yt_true[lx], yt_pred[lx], average='binary')
            MultilabelMetrics._add_prfs(report['labels'][lb], p, r, f1, s)
            MultilabelMetrics._add_tnfpfntp(
                report['labels'][lb], *metrics.confusion_matrix(yt_true[lx], yt_pred[lx]).ravel()
            )
            sum_pos_pred += s

        p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
        MultilabelMetrics._add_prfs(report['avg']['micro'], p, r, f1, sum_pos_pred)
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
        MultilabelMetrics._add_prfs(report['avg']['macro'], p, r, f1, sum_pos_pred)
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        MultilabelMetrics._add_prfs(report['avg']['weighted'], p, r, f1, sum_pos_pred)

        report['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        report['hamming'] = metrics.hamming_loss(y_true, y_pred)
        return report

