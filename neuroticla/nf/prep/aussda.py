import os
import logging
import pandas as pd

from typing import Dict, List
from .filter import DataFilter

logger = logging.getLogger('nf.prep.aussda')


class AussdaDataFilter(DataFilter):

    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def type_mapping(self) -> Dict[str, str]:
        return {
            'headline': 'string',
            'text': 'string',
            'lead_paragraph': 'string',
            'country': 'string',
            'source': 'string',
            'publication_date': 'string'
        }

    def name_mapping(self) -> Dict[str, str]:
        return {
            "publication_date": "published",
            'headline': 'title',
            'lead_paragraph': 'lead',
            'text': 'body',
        }

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        return [
            'headline', 'lead_paragraph', 'text', 'publication_date', 'country', 'source'
        ]

    def required_cols(self) -> List[str]:
        cols = super().required_cols()
        cols.extend([
            'headline', 'text', 'country', 'source'
        ])
        return cols

    def save(self) -> None:
        self.df.to_csv(os.path.join(self.target_dir_path, 'aussda.csv'), index=False)


class AussdaLongDataFilter(AussdaDataFilter):
    # Corpus A includes six countries and 17 media outlets from 2003 to 2017
    # Corpus B covers seven countries and 39 media outlets from 2013 to 2017
    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def type_mapping(self) -> Dict[str, str]:
        tm = super().type_mapping()
        tm['ID'] = 'Int64'
        tm['fr_eco'] = 'Int64'
        tm['fr_lab'] = 'Int64'
        tm['fr_wel'] = 'Int64'
        tm['fr_sec'] = 'Int64'
        tm['fr_cul'] = 'Int64'
        tm['middle_east'] = 'Int64'
        tm['eastern_europe'] = 'Int64'
        tm['filter_corpus_A'] = 'Int64'
        tm['filter_corpus_B'] = 'Int64'
        return tm

    def name_mapping(self) -> Dict[str, str]:
        nm = super().name_mapping()
        nm['ID'] = 'id'
        nm['fr_eco'] = 'eco'
        nm['fr_lab'] = 'lab'
        nm['fr_wel'] = 'wel'
        nm['fr_sec'] = 'sec'
        nm['fr_cul'] = 'cul'
        return nm

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        cols = super().include_cols()
        cols.extend([
            'ID', 'fr_eco', 'fr_lab', 'fr_wel', 'fr_sec', 'fr_cul',
            'middle_east', 'eastern_europe',
            'filter_corpus_A', 'filter_corpus_B'
        ])
        return cols

    def required_cols(self) -> List[str]:
        cols = super().required_cols()
        cols.extend([
            'ID', 'fr_eco', 'fr_lab', 'fr_wel', 'fr_sec', 'fr_cul',
            'middle_east', 'eastern_europe',
            'filter_corpus_A', 'filter_corpus_B'
        ])
        return cols

    def label_cols(self) -> List[str]:
        return ['eco', 'lab', 'wel', 'sec', 'cul']

    def save(self) -> List[str]:
        dfa: pd.DataFrame = self.df[self.df['filter_corpus_A'] == 1]
        logger.info("Got CVS Aussda data size corpus A [%s].", dfa.shape[0])
        dfb: pd.DataFrame = self.df[self.df['filter_corpus_B'] == 1]
        logger.info("Got CVS Aussda data size corpus B [%s].", dfb.shape[0])
        dfab: pd.DataFrame = self.df[(self.df['filter_corpus_A'] == 1) & (self.df['filter_corpus_B'] == 1)]
        logger.info("Got CVS Aussda data size corpus AB [%s].", dfab.shape[0])

        dfa_csv_file = os.path.join(self.target_dir_path, 'aussda_long_a.csv')
        dfa.to_csv(dfa_csv_file, index=False)
        dfb_csv_file = os.path.join(self.target_dir_path, 'aussda_long_b.csv')
        dfb.to_csv(dfb_csv_file, index=False)
        dfab_csv_file = os.path.join(self.target_dir_path, 'aussda_long_ab.csv')
        dfab.to_csv(dfab_csv_file, index=False)
        return [dfa_csv_file, dfb_csv_file, dfab_csv_file]


class AussdaShortDataFilter(AussdaLongDataFilter):
    # Corpus B covers seven countries and 39 media outlets from 2013 to 2017
    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def include_cols(self) -> List[str]:
        cols = super().include_cols()
        cols.remove('filter_corpus_A')
        cols.remove('filter_corpus_B')
        return cols

    def required_cols(self) -> List[str]:
        cols = super().required_cols()
        cols.remove('filter_corpus_A')
        cols.remove('filter_corpus_B')
        return cols

    def label_cols(self) -> List[str]:
        return ['eco', 'lab', 'wel', 'sec', 'cul']

    def save(self) -> List[str]:
        logger.info("Got CVS Aussda short term data size corpus [%s].", self.df.shape[0])
        csv_file = os.path.join(self.target_dir_path, 'aussda_short.csv')
        self.df.to_csv(csv_file, index=False)
        return [csv_file]


class AussdaManualDataFilter(AussdaDataFilter):
    def __init__(self, input_path: str, target_dir_path: str, base_name: str, num_rows: int) -> None:
        super().__init__(input_path, target_dir_path, base_name, num_rows)

    def type_mapping(self) -> Dict[str, str]:
        tm = super().type_mapping()
        tm['reminderid_doc_id'] = 'Int64'
        tm['m_fr_eco'] = 'Int64'
        tm['m_fr_lab'] = 'Int64'
        tm['m_fr_wel'] = 'Int64'
        tm['m_fr_sec'] = 'Int64'
        return tm

    def name_mapping(self) -> Dict[str, str]:
        nm = super().name_mapping()
        nm['reminderid_doc_id'] = 'id'
        nm['m_fr_eco'] = 'eco'
        nm['m_fr_lab'] = 'lab'
        nm['m_fr_wel'] = 'wel'
        nm['m_fr_sec'] = 'sec'
        return nm

    def skip_rows(self, row) -> bool:
        return False

    def include_cols(self) -> List[str]:
        cols = super().include_cols()
        cols.extend([
            'reminderid_doc_id', 'm_fr_eco', 'm_fr_lab', 'm_fr_wel', 'm_fr_sec'
        ])
        return cols

    def required_cols(self) -> List[str]:
        cols = super().required_cols()
        cols.extend([
            'reminderid_doc_id', 'm_fr_eco', 'm_fr_lab', 'm_fr_wel', 'm_fr_sec'
        ])
        return cols

    def label_cols(self) -> List[str]:
        return ['eco', 'lab', 'wel', 'sec']

    def save(self) -> List[str]:
        logger.info("Got CVS Aussda manual data size corpus [%s].", self.df.shape[0])
        csv_file = os.path.join(self.target_dir_path, 'aussda_manual.csv')
        self.df.to_csv(csv_file, index=False)
        return [csv_file]


# Spain
#   Print:  ABC, El Mundo, El Pais
# UK
#   Print:  Daily Mail, Daily Mirror, Metro, The Daily Telegraph, The Guardian
#   Online: mirror.co.uk, telegraph.co.uk
# Germany
#   Print:  Bild, Die Tageszeitung (taz), Frankfurter Allgemeine Zeitung, Frankfurter Rundschau, Süddeutsche Zeitung
#   Online: spiegel.de, sueddeutsche.de, welt.de, zeit.de
# Sweden
#   Print: Aftonbladet, Dagens Nyheter, Expressen, Svenska Dagbladet
# Poland
#   Print:  Dziennik Gazeta Prawna, Gazeta Wyborcza, Rzeczpospolita
#   Online: gazeta.pl
# Hungary
#   Print:  Magyar Hirlap, Magyar Idök, Nepszabadsag, Nepszava
#   Online: 24.hu, blikk.hu, magyarhirlap.hu, napi.hu, nepszava.hu
# Romania
#   Print:  Evenimentul Zilei, Jurnalul National, Romania Libera, Ziarul Financiar

# Migration relevant?
# Language	Man. anno.	Man. relevant.	Search app	Recall	 Precision
# Spanish	2,113	    104	            105	        0.92	   0.93
# English	3,418	    170	            165	        0.88	   0.9
# German	1,203	    119	            111	        0.89	   0.94
# Swedish	1,244	    85	            60	        0.67	   0.93
# Polish	1,391	    63	            63	        0.77	   0.76
# Hungarian 1,200	    102	            101	        0.83	   0.81
# Romanian  1,415	    63	            61	        0.71	   0.71


# Frame	    Man anno	Man relevant	Dict app	Recall	Precision
# Economy	2,000	    434	            437	        0.82	  0.81
# Labour    2,000	    585	            501	        0.69	  0.81
# Welfare	2,000	    461	            427	        0.69	  0.82
# Security	2,000	    848	            823	        0.84	  0.86
# Culture	2,000	    665	            690	        0.86	  0.83


# long
# ['ID', 'origin_ID', 'sample_ID', 'article_id', 'country', 'source',
#  'source_type', 'publication_date', 'filter_corpus_A', 'filter_corpus_B',
#  'weight', 'total_coverage', 'european_mobil', 'sentiment_mr_sentences',
#  'sentiment_other_sentences', 'sentiment_overall', 'fr_eco', 'fr_lab',
#  'fr_wel', 'fr_sec', 'fr_cul', 'fr_eco_sent', 'fr_lab_sent',
#  'fr_wel_sent', 'fr_sec_sent', 'fr_cul_sent', 'middle_east',
#  'eastern_europe', 'author', 'section', 'headline', 'lead_paragraph',
#  'text', 'headline_mt', 'lead_paragraph_mt', 'text_mt'
#
#  short
#  ['ID', 'ID_origin', 'ID_sample', 'ID_aussda', 'country', 'source_type',
#  'source', 'publication_date', 'publication_time', 'author', 'section',
#  'total_coverage', 'european_mobil', 'sentiment_mr_sentences',
#  'sentiment_other_sentences', 'sentiment_overall', 'fr_eco', 'fr_lab',
#  'fr_wel', 'fr_sec', 'fr_cul', 'fr_eco_sent', 'fr_lab_sent',
#  'fr_wel_sent', 'fr_sec_sent', 'fr_cul_sent', 'middle_east',
#  'eastern_europe', 'headline', 'lead_paragraph', 'text', 'headline_mt',
#  'lead_paragraph_mt', 'text_mt'
#
#  manual
#  ['reminderid_doc_id', 'country', 'publication_date', 'source',
#  'source_type', 'm_fr_eco', 'm_fr_lab', 'm_fr_wel', 'm_fr_sec',
#  'headline', 'lead_paragraph', 'text', 'headline_mt',
#  'lead_paragraph_mt', 'text_mt', 'all_text_orig_lang', 'all_text_mt',
#  'all_text_orig_lang_lemma', 'all_text_mt_lemma', 'm_fr_eco_exclusive',
#  'm_fr_lab_exclusive', 'm_fr_wel_exclusive', 'm_fr_sec_exclusive']
