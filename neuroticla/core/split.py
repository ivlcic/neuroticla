import os
import logging
import numpy as np
import pandas as pd

from typing import Dict, List
logger = logging.getLogger('neuroticla.core.split')


class DataSplit:

    @classmethod
    def split_data(cls, args, base_name: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        ds = [int(x) for x in args.data_split.split(':')]
        if len(ds) != 2:
            raise ValueError("We need two-way split arg: 'train:eval' we use remains as 'test' "
                             "i.e. split arg '80:15' would leave us with 5% test set size")
        if sum(ds) >= 100:
            raise ValueError("Data split sum must be less than 100 since we also need a test set!")

        data = pd.read_csv(os.path.join(args.data_dir, base_name + '.csv'))

        # Shuffle the whole dataset first
        if args.non_reproducible_shuffle:
            data = data.sample(frac=1).reset_index()
            logger.info("Done non-reproducible data shuffle.")
        else:
            data = data.sample(frac=1, random_state=2611).reset_index()
            logger.info("Done reproducible data shuffle.")

        data_len = len(data)
        train_n = int((ds[0] / 100) * data_len)
        eval_n = train_n + int((ds[1] / 100) * data_len)
        training_data, evaluation_data, test_data = np.split(data, [train_n, eval_n])
        logger.info("Data split [%s:%s] proportionally to [%s] => [train:%s,eval:%s,test:%s]",
                    base_name, data_len, args.data_split,
                    len(training_data), len(evaluation_data), len(test_data))

        training_data.to_csv(
            os.path.join(args.data_dir, base_name + '.train.csv'), index=False, encoding='utf-8'
        )

        evaluation_data.to_csv(
            os.path.join(args.data_dir, base_name + '.eval.csv'), index=False, encoding='utf-8'
        )

        test_data.to_csv(
            os.path.join(args.data_dir, base_name + '.test.csv'), index=False, encoding='utf-8'
        )

        return training_data, evaluation_data, test_data

    @classmethod
    def multi_split_data(cls, args, confs: List[Dict]) -> None:
        ds = [int(x) for x in args.data_split.split(':')]
        if len(ds) != 2:
            raise ValueError("We need two-way split arg: 'train:eval' we use remains as 'test' "
                             "i.e. split arg '80:15' would leave us with 5% test set size")
        if sum(ds) >= 100:
            raise ValueError("Data split sum must be less than 100 since we also need a test set!")

        training_sets: List[pd.DataFrame] = []
        evaluation_sets: List[pd.DataFrame] = []
        test_sets: List[pd.DataFrame] = []
        for conf in confs:
            # target_base_name = nf.args.chech_param(conf, 'result_name')
            training_data, evaluation_data, test_data = cls.split_data(args, target_base_name)
            training_sets.append(training_data)
            evaluation_sets.append(evaluation_data)
            test_sets.append(test_data)

        pd.concat(training_sets).to_csv(
            os.path.join(args.data_dir, args.lang + '.train.csv'), index=False, encoding='utf-8'
        )
        pd.concat(evaluation_sets).to_csv(
            os.path.join(args.data_dir, args.lang + '.eval.csv'), index=False, encoding='utf-8'
        )
        pd.concat(test_sets).to_csv(
            os.path.join(args.data_dir, args.lang + '.test.csv'), index=False, encoding='utf-8'
        )
