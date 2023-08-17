import os
import logging

from typing import List
from .filter import DataFilter

logger = logging.getLogger('nf.prep.slomcor')


class SlomcorDataFilter(DataFilter):

    def __init__(self, args) -> None:
        super().__init__(args)

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

    def save(self) -> None:
        if 'middle' in self.args.input_file:
            logger.info("Got CVS Slomcor Middle-East data size corpus [%s].", len(self.df.index))
            self.df.to_csv(os.path.join(self.args.data_out_dir, 'slomcor_middle_east.csv'), index=False)
        else:
            logger.info("Got CVS Slomcor Ukraine data size corpus [%s].", len(self.df.index))
            self.df.to_csv(os.path.join(self.args.data_out_dir, 'slomcor_ukraine.csv'), index=False)
