import os
import logging
import numpy as np
import pandas as pd

from typing import Dict, List
logger = logging.getLogger('core.split')


class DataSplit:

    @classmethod
    def split(cls, data_split: str, base_name: str, file_set: List[str],
              random_state: int = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        ds = [int(x) for x in data_split.split(':')]
        if len(ds) != 2:
            raise ValueError("We need two-way split arg: 'train:eval' we use remains as 'test' "
                             "i.e. split arg '80:15' would leave us with 5% test set size")
        if sum(ds) >= 100:
            raise ValueError("Data split sum must be less than 100 since we also need a test set!")

        frames: List[pd.DataFrame] = []
        for f in file_set:
            df: pd.DataFrame = pd.read_csv(f, encoding='utf-8')
            frames.append(df)
        data: pd.DataFrame = pd.concat(frames)
        if random_state is not None:
            data = data.sample(frac=1, random_state=random_state)
            logger.info("Done reproducible data shuffle with random state: %s.", random_state)
        else:
            data = data.sample(frac=1)
            logger.info("Done non-reproducible data shuffle.")

        train_n = int((ds[0] / 100) * data.shape[0])
        eval_n = train_n + int((ds[1] / 100) * data.shape[0])
        training_data, evaluation_data, test_data = np.split(data, [train_n, eval_n])
        logger.info(
            "Data split [%s:%s] proportionally to [%s] => [train:%s,eval:%s,test:%s]",
            base_name, data.shape[0], data_split,
            training_data.shape[0], evaluation_data.shape[0], test_data.shape[0]
        )
        return training_data, evaluation_data, test_data

    @classmethod
    def multi_split(cls, data_split: str, file_sets: Dict[str, List[str]],
                    random_state: int = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        training_sets: List[pd.DataFrame] = []
        evaluation_sets: List[pd.DataFrame] = []
        test_sets: List[pd.DataFrame] = []
        for k, file_set in file_sets.items():
            training_data, evaluation_data, test_data = cls.split(data_split, k, file_set, random_state)
            training_sets.append(training_data)
            evaluation_sets.append(evaluation_data)
            test_sets.append(test_data)

        training_df = pd.concat(training_sets)
        evaluation_df = pd.concat(evaluation_sets)
        test_df = pd.concat(test_sets)
        return training_df, evaluation_df, test_df

    @classmethod
    def load(cls, path_prefixes: List[str]) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

        training_sets: List[pd.DataFrame] = []
        evaluation_sets: List[pd.DataFrame] = []
        test_sets: List[pd.DataFrame] = []
        for path_prefix in path_prefixes:
            logger.debug("Loading corpus [%s]...", path_prefix)
            if not os.path.exists(path_prefix + '.train.csv'):
                delim = '_'
            else:
                delim = '.'
            training_sets.append(
                pd.read_csv(path_prefix + delim + 'train.csv')
            )
            evaluation_sets.append(
                pd.read_csv(path_prefix + delim + 'eval.csv')
            )
            test_sets.append(
                pd.read_csv(path_prefix + delim + 'test.csv')
            )
            logger.info("Loaded corpus [%s]", path_prefix)

        return pd.concat(training_sets), pd.concat(evaluation_sets), pd.concat(test_sets)
