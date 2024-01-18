import os
import json
import logging

from datetime import datetime, timedelta
from typing import Callable, Any, Dict

logger = logging.getLogger('play.cluster.dump')


class State:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end
        self.currentDate: datetime
        self.index: int = 0
        self.size: int = 0
        self.total: int = 0
        self.file: str = ''
        self.relPath: str = ''
        self.log: Dict[str, Any] = {}


load_range_callback = Callable[[State, Dict[str, Any]], int]


def load_range(start_date: str, end_date: str, result_dir: str, callback: load_range_callback) -> State:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    state: State = State(start, end)
    current_date = end
    while current_date > start:
        prev_day = current_date - timedelta(days=1)
        rel_path = os.path.join(str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}")
        day_dir = os.path.join(result_dir, rel_path)
        if os.path.exists(day_dir):
            file_names = os.listdir(day_dir)
            state.currentDate = current_date
            state.size = len(file_names)
            for x, article_file in enumerate(file_names):
                if not article_file.endswith('.json'):
                    continue
                article_file = os.path.join(day_dir, article_file)
                if not os.path.exists(article_file):
                    continue
                with open(article_file, encoding='utf-8') as json_file:
                    try:
                        saved_article = json.load(json_file)
                    except:
                        logger.error("Unable to load json file [%s].", article_file)
                        os.remove(article_file)
                        return
                    state.index = x
                    state.article = saved_article
                    state.file = article_file
                    state.relPath = day_dir

                    state.total += callback(state, saved_article)

        logger.info("Finished interval [%s::%s] %s",prev_day, current_date, state.log)
        current_date = prev_day
    return state