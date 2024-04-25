import os
import json
import numpy as np
import pandas as pd

from typing import Dict, Any, Union, List

from neuroticla.core.labels import Labeler


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ResultsCollector:

    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self._y_pred_all = []
        self._y_true_all = []

    def collect(self, labeler: Union[Labeler, None], y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self._y_true_all.append(y_true)
        self._y_pred_all.append(y_pred)

    def get_all_true(self) -> np.ndarray:
        return np.array(self._y_true_all).transpose().tolist()

    def get_all_pred(self) -> np.ndarray:
        return np.array(self._y_pred_all).transpose().tolist()


class ResultWriterOld:

    def __init__(self, result_dir: str, model_dir: str, total_name: Union[str, None] = 'results_all'):
        self._result_dir = result_dir
        self._model_dir = model_dir
        self._total_name = total_name

    def write(self, results: Dict[str, Any], model_name: str, r_base_name: str = None):
        combined_results = {}
        if self._total_name is not None:
            total_path = os.path.join(self._result_dir, self._total_name + '.json')
            if os.path.exists(total_path):
                with open(total_path) as json_file:
                    combined_results = json.load(json_file)

            combined_results[model_name] = results
            with open(total_path, 'wt', encoding='utf-8') as fp:
                json.dump(combined_results, fp, cls=NpEncoder, indent=2)

        f_name = os.path.join(self._model_dir, model_name + ".json") if r_base_name is None \
            else os.path.join(self._model_dir, r_base_name + ".json")
        with open(f_name, 'wt') as fp:
            json.dump(combined_results, fp, cls=NpEncoder, indent=2)


class ResultWriter:

    def __init__(self):
        pass

    def flatten(self, current, key, result):
        if isinstance(current, dict):
            for k in current:
                new_key = "{0}.{1}".format(key, k) if len(key) > 0 else k
                self.flatten(current[k], new_key, result)
        else:
            result[key] = current
        return result

    def write_predictions(self, path: str, base_name: str, data: pd.DataFrame, drop: List[str] = []):
        checked_drop = []
        for d in drop:
            if d in data:
                checked_drop.append(d)
        tmp_data = data.drop(checked_drop, axis=1)
        file_path = os.path.join(path, base_name + '.cvs')
        tmp_data.to_csv(file_path, encoding='utf-8', index=False)

    def write_metrics(self, path: str, base_name: str, model_name: str, results: Dict[str, Any],
                      overwrite: bool = False):
        total_path = os.path.join(path, base_name + '.json')
        combined_results = {}
        if os.path.exists(total_path) and not overwrite:
            with open(total_path) as json_file:
                combined_results = json.load(json_file)

        combined_results[model_name] = results
        with open(total_path, 'wt', encoding='utf-8') as fp:
            json.dump(combined_results, fp, cls=NpEncoder, indent=2)

        results = []
        for m_name, model_data in combined_results.items():
            model_data['model_name'] = m_name
            model_result = self.flatten(model_data, '', {})
            results.append(model_result)

        d = pd.DataFrame(results)
        d.to_csv(os.path.join(path, base_name + '.csv'))
