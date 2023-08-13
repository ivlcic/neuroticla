class DataFilter:

    @classmethod
    def get(cls) -> DataFilter:

        pass

    def __init__(self) -> None:
        super().__init__()

    def filter(self) -> None:
        pass


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