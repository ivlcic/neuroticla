import os
import json
from typing import Dict, Any

import numpy as np


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ResultWriter:

    def __init__(self, result_dir: str, total_name: str = 'results_all'):
        self._result_dir = result_dir
        self._total_name = total_name

    def write(self, results: Dict[str, Any], model_name):
        combined_results = {}
        total_path = os.path.join(self._result_dir, self._total_name + '.json')
        if os.path.exists(total_path):
            with open(total_path) as json_file:
                combined_results = json.load(json_file)

        combined_results[model_name] = results
        with open(total_path, 'wt', encoding='utf-8') as fp:
            json.dump(combined_results, fp, cls=NpEncoder)
        with open(os.path.join(self._result_dir, model_name + ".json"), 'wt') as fp:
            json.dump(results, fp, cls=NpEncoder)
