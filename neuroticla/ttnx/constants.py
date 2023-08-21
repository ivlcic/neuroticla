TTNX_API_KEY = 'TTNX_API_KEY'

TTNX_AVG_SQUEEZE = "squeeze"
TTNX_AVG_SENTENCE = "sentence"
TTNX_AVG_TRUNCATE = "truncate"
TTNX_AVG_NONE = "none"
TTNX_WEIGHT_NEG_LIN = "neg_lin"
TTNX_WEIGHT_NEG_EXP = "neg_exp"

TTNX_DEFAULT_AVG = TTNX_AVG_SQUEEZE
TTNX_DEFAULT_WEIGHT = TTNX_WEIGHT_NEG_LIN


MODEL_CACHE_DIR='tmp'

LOCAL_AVG_SENT = 1
LOCAL_AVG_NONE = 0
LOCAL_AVG_SQUEEZE = 2
LOCAL_AVG_TRUNCATE = 3

LOCAL_WEIGHT_NONE = 0
LOCAL_WEIGHT_NEG_LIN = 1
LOCAL_WEIGHT_NEG_EXP = 2

MODEL_NAME_MAP = {
    'mcbert': 'bert-base-multilingual-cased',
    'xlmrb': 'xlm-roberta-base',
    'xlmrl': 'xlm-roberta-large',
    'st.para': 'paraphrase-multilingual-mpnet-base-v2',
    'st.mpnet': 'all-mpnet-base-v2'
}
