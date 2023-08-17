import re
import os
import logging
import pandas as pd

from typing import Dict, List, Union, Any


logger = logging.getLogger('nf.prep.filter')


class DataFilter:
    countries = {
        'Hungary': 'hu',
        'Spain': 'es',
        'Sweden': 'se',
        'UK': 'uk',
        'Bulgaria': 'bg',
        'Germany': 'de',
        'Romania': 'ro',
        'Poland': 'pl'
    }

    languages = {
        'Hungary': 'hu',
        'Spain': 'es',
        'Sweden': 'sv',
        'UK': 'en',
        'Bulgaria': 'bg',
        'Germany': 'de',
        'Romania': 'ro',
        'Poland': 'pl'
    }

    @classmethod
    def cleanup_text(cls, s: str):
        if not isinstance(s, str):
            return
        s = s.replace(' ', ' ').replace('\t', ' ').replace('\r', '')
        s = re.sub('"+', '"', s)
        s = re.sub(' +\n+', '\n', s)
        s = re.sub('\n+', '\n', s)
        s = re.sub(' +', ' ', s)
        return s.strip()

    @classmethod
    def filter_country(cls, s: str):
        if not isinstance(s, str):
            return
        return cls.countries[s]

    @classmethod
    def filter_language(cls, s: str):
        if not isinstance(s, str):
            return
        return cls.languages[s]

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.num_rows = args.num_rows
        self.df: Union[pd.DataFrame, Any] = None

    def type_mapping(self) -> Dict[str, str]:
        return {}

    def name_mapping(self) -> Dict[str, str]:
        return {}

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        return []

    def required_cols(self) -> List[str]:
        return []

    def load(self) -> None:
        tm = self.type_mapping()
        ic = self.include_cols()
        rc = self.required_cols()
        self.df: pd.DataFrame = pd.read_csv(
            self.args.input_path,
            dtype=tm if tm else None,
            skiprows=lambda x: self.skip_rows(x),
            usecols=ic if ic else None,
            encoding='utf-8',
            nrows=self.args.num_rows
        )
        logger.info("Got CVS data size [%s] after loading.", self.df.size)
        nm = self.name_mapping()
        self.df.dropna(
            subset=rc, inplace=True
        )
        logger.info("Got CVS data size [%s] after dropping.", self.df.size)
        if nm:
            self.df.rename(columns=nm, inplace=True)
        logger.info(
            "Got CVS data size [%s] columns after first filtering: %s", self.df.size, self.df.columns
        )

    def filter(self) -> None:
        for i, row in self.df.iterrows():
            body: str = DataFilter.cleanup_text(self.df.at[i, 'body'])
            title: str = DataFilter.cleanup_text(self.df.at[i, 'title'])
            lead: str = DataFilter.cleanup_text(self.df.at[i, 'lead'])
            country: str = DataFilter.filter_country(self.df.at[i, 'country'])
            if body is not None:
                if body.startswith(title):
                    body = body[len(title):]
                self.df.at[i, 'body'] = body
            if lead is not None:
                if lead.startswith(title):
                    lead = lead[len(title):]
                self.df.at[i, 'lead'] = lead
            if country is not None:
                self.df.at[i, 'country'] = country
            self.df.at[i, 'title'] = title

    def save(self) -> None:
        self.df.to_csv(os.path.join(self.args.data_out_dir, 'filtered.csv'))
