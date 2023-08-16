import pandas as pd

from typing import Dict, List


class DataFilter:

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

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.num_rows = args.num_rows
        self.df: pd.DataFrame = None

    def type_mapping(self) -> Dict[str, str]:
        return {}

    def name_mapping(self) -> Dict[str, str]:
        return {}

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        return []

    def load(self) -> None:
        tm = self.type_mapping()
        ic = self.include_cols()
        self.df: pd.DataFrame = pd.read_csv(
            self.args.input_path,
            dtype=tm if tm else None,
            skiprows=lambda x: self.skip_rows(x),
            usecols=ic if ic else None,
            encoding='utf-8',
            nrows=self.args.num_rows
        )
        nm = self.name_mapping()
        if nm:
            df.rename(columns=nm, inplace=True)

    def filter(self) -> None:
        for i, row in self.df.iterrows():
            body: str = DataFilter.cleanup_text(self.df.at[i, 'body'])
            title: str = DataFilter.cleanup_text(self.df.at[i, 'title'])
            lead: str = DataFilter.cleanup_text(self.df.at[i, 'lead'])
            if body.startswith(title):
                body = body[len(title):]
            self.df.at[i, 'body'] = body
            self.df.at[i, 'title'] = title
            self.df.at[i, 'lead'] = lead

    def save(self) -> None:
        self.df.to_csv(os.path.join(self.args.data_out_dir, 'tmp.csv'))
