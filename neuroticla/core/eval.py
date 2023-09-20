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

    @classmethod
    def _compute_prf1(cls, tn, fp, fn, tp):
        if tp == 0:
            return 0, 0, 0
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        return p, r, f1

    def compute_old(self, references: Union[List[Any], np.ndarray], predictions: Union[List[Any], np.ndarray],
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

    def compute(self, references: Union[List[Any], np.ndarray], predictions: Union[List[Any], np.ndarray],
                labels: List[str], output_dict: bool = True):
        # init container dict
        report = {'labels': {}, 'avg': {}}
        y_true: np.ndarray = np.array(references)
        y_pred: np.ndarray = np.array(predictions)

        if len(labels) == 1:
            # for single label we get  per label row vector
            yt_t = np.array([y_true])
            yt_p = np.array([y_pred])
        else:
            # we transform column-per-label (1-hot encoded matrix or label indicator array) to row-per-label vector
            yt_t = y_true.transpose()
            yt_p = y_pred.transpose()

        # per label evaluation decomposition
        sum_prf1s_0 = np.array([0, 0, 0, 0], dtype='float64')
        sum_prf1s_1 = np.array([0, 0, 0, 0], dtype='float64')
        sum_cm_0 = np.array([[0, 0], [0, 0]])
        sum_cm_1 = np.array([[0, 0], [0, 0]])
        for lx, lb in enumerate(labels):
            report['labels'][lb + '-0'] = {}
            report['labels'][lb + '-1'] = {}
            # note the absence of an average parameter
            p, r, f1, s = metrics.precision_recall_fscore_support(yt_t[lx], yt_p[lx], zero_division=0)
            MultilabelMetrics._add_prfs(report['labels'][lb + '-0'], p[0], r[0], f1[0], s[0])
            MultilabelMetrics._add_prfs(report['labels'][lb + '-1'], p[1], r[1], f1[1], s[1])
            cm_1 = metrics.confusion_matrix(yt_t[lx], yt_p[lx])
            # negative label CM is just pi rotation
            cm_0 = np.rot90(np.rot90(cm_1))
            MultilabelMetrics._add_tnfpfntp(report['labels'][lb + '-0'], *cm_0.ravel())
            MultilabelMetrics._add_tnfpfntp(report['labels'][lb + '-1'], *cm_1.ravel())
            sum_prf1s_0 += [p[0], r[0], f1[0], s[0]]
            sum_prf1s_1 += [p[1], r[1], f1[1], s[1]]
            sum_cm_0 += cm_0
            sum_cm_1 += cm_1

        l_len = len(labels)
        l_2len = 2 * l_len

        report['avg'] = {
            'macro-0': {}, 'macro-1': {}, 'macro': {},
            'micro-0': {}, 'micro-1': {}, 'micro': {}
        }

        # macro averaging
        sum_prf1s = sum_prf1s_0 + sum_prf1s_1
        MultilabelMetrics._add_prfs(
            report['avg']['macro-0'], *np.divide(sum_prf1s_0, [l_len, l_len, l_len, 1]).ravel()
        )
        MultilabelMetrics._add_prfs(
            report['avg']['macro-1'], *np.divide(sum_prf1s_1, [l_len, l_len, l_len, 1]).ravel()
        )
        MultilabelMetrics._add_prfs(
            report['avg']['macro'], *np.divide(sum_prf1s, [l_2len, l_2len, l_2len, 1]).ravel()
        )

        # micro averaging
        sum_cm = sum_cm_0 + sum_cm_1
        MultilabelMetrics._add_prfs(
            report['avg']['micro-0'], *MultilabelMetrics._compute_prf1(*sum_cm_0.ravel()), sum_prf1s_0[3]
        )
        MultilabelMetrics._add_tnfpfntp(report['avg']['micro-0'], *sum_cm_0.ravel())
        MultilabelMetrics._add_prfs(
            report['avg']['micro-1'], *MultilabelMetrics._compute_prf1(*sum_cm_1.ravel()), sum_prf1s_1[3]
        )
        MultilabelMetrics._add_tnfpfntp(report['avg']['micro-1'], *sum_cm_1.ravel())
        MultilabelMetrics._add_prfs(
            report['avg']['micro'], *MultilabelMetrics._compute_prf1(*sum_cm.ravel()), sum_prf1s[3]
        )
        MultilabelMetrics._add_tnfpfntp(report['avg']['micro'], *sum_cm.ravel())

        report['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        report['hamming'] = metrics.hamming_loss(y_true, y_pred)
        return report
