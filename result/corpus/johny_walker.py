import os
import json
import logging

from datetime import datetime, timedelta
from typing import Union, Dict, Any, Callable

logger = logging.getLogger('johny.walker')


class State:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end
        self.current: Union[datetime, None] = None
        self.current_idx: int = 0
        self.current_size: int = 0
        self.total: int = 0
        self.file: str = ''
        self.rel_path: str = ''
        self.log: Dict[str, Any] = {}


class Params:
    def __init__(self, start_date: str, end_date: str, corpus_path: str):
        self.start = datetime.fromisoformat(start_date)
        self.end = datetime.fromisoformat(end_date)
        self.corpus_path = corpus_path


item_callback = Callable[[State, Dict[str, Any]], int]


def walk_range(params: Params, callback: item_callback) -> State:
    state: State = State(params.start, params.end)
    current_date = params.end
    while current_date > params.start:
        prev_day = current_date - timedelta(days=1)
        state.rel_path = os.path.join(str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}")
        day_dir = os.path.join(params.corpus_path, state.rel_path)
        state.current = current_date
        file_names = os.listdir(day_dir)
        state.current_size = len(file_names)
        for x, article_file in enumerate(file_names):
            if not article_file.endswith('.json'):
                continue
            article_file = os.path.join(day_dir, article_file)
            if not os.path.exists(article_file):
                continue
            with open(article_file, encoding='utf-8') as json_file:
                # noinspection PyBroadException
                try:
                    saved_article = json.load(json_file)
                except Exception:
                    logger.error("Unable to load json file [%s].", article_file)
                    os.remove(article_file)
                    return state
                state.current_idx = x
                state.file = article_file
                state.total += callback(state, saved_article)

        logger.info("Finished interval [%s::%s] %s", prev_day, current_date, state.log)
        current_date = prev_day
    return state
