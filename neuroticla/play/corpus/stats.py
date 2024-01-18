import os
import logging
import csv
from typing import Dict

import numpy as np
import pandas as pd

from neuroticla.play.corpus.__utils import load_range, State

logger = logging.getLogger('play.cluster.stats')


def _calculate_percentiles(data, percentiles=None):
    if percentiles is None:
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    return {metric: np.percentile(data[metric], percentiles) for metric in data}


def collect(arg) -> int:
    data = {
        'uuid': [],
        'published': [],
        'characters': [],
        'sentences': [],
        'word_tok': [],
        'sp_tok': [],
        'cl100k_tok': [],
        'country': [],
        'language': [],
        'media_type': [],
        'industries': [],
        'rel_path': [],
        'title': []
    }
    industry_map = {}
    map_file_name = os.path.join(arg.result_dir, 'tag_industries_map.csv')
    with open(map_file_name, encoding='utf-8') as map_file:
        try:
            reader = csv.reader(map_file)
            for row in reader:
                # Assuming the first column is the key and the second is the value
                key = row[0]
                value = row[1]
                industry_map[key] = value
        except:
            logger.error("Unable to load CSV tag map file [%s].", map_file_name)
            return 1
    #  distinct_industries = set(industry_map.values())

    def callback(s: State, saved_article: Dict) -> int:
        data['uuid'].append(saved_article['uuid'])
        data['published'].append(saved_article['published'])
        data['characters'].append(saved_article['stats']['chr'])
        data['sentences'].append(saved_article['stats']['sent'])
        data['word_tok'].append(saved_article['stats']['w_t'])
        data['sp_tok'].append(saved_article['stats']['sp_t'])
        data['cl100k_tok'].append(saved_article['stats']['oai_t'])
        data['country'].append(saved_article['country']['name'])
        data['language'].append(saved_article['language'].split('-', 1)[0])
        data['media_type'].append(saved_article['media']['type']['name'])

        ids = set([industry_map[t['uuid']] for t in saved_article['tags'] if t['uuid'] in industry_map])
        data['industries'].append(ids)
        data['rel_path'].append(s.relPath)
        data['title'].append(saved_article['title']['text'])
        return 1

    state = load_range(arg.start_date, arg.end_date, arg.result_dir, callback)

    logger.info(
        "Collected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    # Calculate percentiles
    # percentiles = _calculate_percentiles(data)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(arg.result_dir, f'stats-{arg.start_date}_{arg.end_date}.csv'))
    return 0
