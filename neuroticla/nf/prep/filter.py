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
        'Poland': 'pl',
        'SI': 'si'
    }

    languages = {
        'Hungary': 'hu',
        'Spain': 'es',
        'Sweden': 'sv',
        'UK': 'en',
        'Bulgaria': 'bg',
        'Germany': 'de',
        'Romania': 'ro',
        'Poland': 'pl',
        'SI': 'sl'
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

    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__()
        self.input_path = input_path
        self.target_dir_path = target_dir_path
        self.num_rows = num_rows
        self.base_name = base_name
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

    def label_cols(self) -> List[str]:
        return []

    def load(self) -> None:
        tm = self.type_mapping()
        ic = self.include_cols()
        rc = self.required_cols()
        self.df: pd.DataFrame = pd.read_csv(
            self.input_path,
            dtype=tm if tm else None,
            skiprows=lambda x: self.skip_rows(x),
            usecols=ic if ic else None,
            encoding='utf-8',
            nrows=self.num_rows
        )
        logger.info(
            "Got CVS data size [%s] after loading with included cols %s.",
            self.df.shape[0], ic
        )
        nm = self.name_mapping()
        self.df.dropna(
            subset=rc, inplace=True
        )
        logger.info(
            "Got CVS data size [%s] after dropping samples missing required cols %s.",
            self.df.shape[0], rc
        )
        if nm:
            self.df.rename(columns=nm, inplace=True)
        logger.info(
            "Renamed CVS data of size [%s] with mapping %s.",
            self.df.shape[0], nm
        )
        self.df.reindex()
        logger.info(
            "Got CVS data size [%s] columns after first filtering and renaming: %s",
            self.df.shape[0], self.df.columns
        )

    def filter(self) -> None:
        for i, row in self.df.iterrows():
            body: str = DataFilter.cleanup_text(self.df.at[i, 'body'])
            title: str = DataFilter.cleanup_text(self.df.at[i, 'title'])
            country: str = DataFilter.filter_country(self.df.at[i, 'country'])
            if body:
                if body.startswith(title):
                    body = body[len(title):]
                self.df.at[i, 'body'] = body
            if country:
                self.df.at[i, 'country'] = country
            self.df.at[i, 'title'] = title

            if 'lead' in self.df:
                lead: str = DataFilter.cleanup_text(self.df.at[i, 'lead'])
                if lead:
                    if lead.startswith(title):
                        lead = lead[len(title):]
                    self.df.at[i, 'lead'] = lead

    def save(self) -> List[str]:
        csv_file = os.path.join(self.target_dir_path, 'filtered.csv')
        self.df.to_csv(csv_file, index=False)
        return [csv_file]

    def data(self) -> pd.DataFrame:
        return self.df
