import os
import logging

from typing import List, Dict
from .filter import DataFilter

logger = logging.getLogger('nf.prep.slomcor')


class SlomcorDataFilter(DataFilter):

    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        return [
            'id', 'title', 'body', 'published', 'country', 'source'
        ]

    def required_cols(self) -> List[str]:
        return [
            'id', 'title', 'body', 'published', 'country', 'source'
        ]

    def save(self) -> List[str]:
        base_name = os.path.basename(self.input_path)
        logger.info("Got CVS Slomcor [%s] data size corpus [%s].", base_name, self.df.shape[0])
        csv_file = os.path.join(self.target_dir_path, base_name)
        self.df.to_csv(csv_file, index=False)
        return [csv_file]


class SlomcorManualDataFilter(DataFilter):

    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def skip_rows(self, row) -> bool:
        return False

    def type_mapping(self) -> Dict[str, str]:
        tm = super().type_mapping()
        tm['id'] = 'Int64'
        tm['fr_eco'] = 'Int64'
        tm['fr_lab'] = 'Int64'
        tm['fr_wel'] = 'Int64'
        tm['fr_sec'] = 'Int64'
        tm['fr_cul'] = 'Int64'
        tm['migration'] = 'Int64'
        return tm

    def name_mapping(self) -> Dict[str, str]:
        nm = super().name_mapping()
        nm['fr_eco'] = 'eco'
        nm['fr_lab'] = 'lab'
        nm['fr_wel'] = 'wel'
        nm['fr_sec'] = 'sec'
        nm['fr_cul'] = 'cul'
        return nm

    def include_cols(self) -> List[str]:
        return [
            'id', 'title', 'body', 'published', 'country', 'source', 'migration',
            'fr_eco', 'fr_lab', 'fr_wel', 'fr_sec', 'fr_cul'
        ]

    def required_cols(self) -> List[str]:
        return [
            'id', 'title', 'body', 'published', 'country', 'source', 'migration',
            'fr_eco', 'fr_lab', 'fr_wel', 'fr_sec', 'fr_cul'
        ]

    def save(self) -> List[str]:
        base_name = os.path.basename(self.input_path)
        logger.info("Got CVS Slomcor manual [%s] data size corpus [%s].", base_name, self.df.shape[0])
        csv_file = os.path.join(self.target_dir_path, base_name)
        self.df.to_csv(csv_file, index=False)
        return [csv_file]
