import os
import logging

from typing import List
from .filter import DataFilter

logger = logging.getLogger('nf.prep.slomcor')


class SlomcorDataFilter(DataFilter):

    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        return [
            'title', 'body', 'published', 'country', 'source'
        ]

    def required_cols(self) -> List[str]:
        return [
            'title', 'body', 'published', 'country', 'source'
        ]

    def save(self) -> List[str]:
        base_name = os.path.basename(self.input_path)
        logger.info("Got CVS Slomcor [%s] data size corpus [%s].", base_name, self.df.shape[0])
        csv_file = os.path.join(self.target_dir_path, base_name)
        self.df.to_csv(csv_file, index=False)
        return [csv_file]
