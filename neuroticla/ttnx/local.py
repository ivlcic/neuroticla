import logging
import math

import torch
import obeliks
import numpy as np
import sentence_transformers as st

from typing import List, Union
from numpy import ndarray
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding

from ..esdl.article import Article
from .constants import MODEL_CACHE_DIR, LOCAL_AVG_SQUEEZE, LOCAL_WEIGHT_NEG_EXP, LOCAL_WEIGHT_NEG_LIN, \
    MODEL_NAME_MAP, LOCAL_WEIGHT_NONE

logger = logging.getLogger('local.embed')


class LocalModel:

    def __init__(self, model_name_or_path, cache_model_dir):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_model_dir
        )
        self._model: PreTrainedModel = AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_model_dir
        )
        self._model.to(device)
        self._model.eval()

    def get_max_seq_length(self) -> int:
        return self._tokenizer.max_len_single_sentence

    def tokenize(self, text: List[str]):
        model_inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=False,
        )
        return model_inputs

    def encode(self, text: Union[str, List[str]], convert_to_tensor=True, show_progress_bar=False):
        model_inputs: BatchEncoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=False,
        )
        model_inputs.to(self._device)
        outputs = self._model(**model_inputs)
        text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach()
        return text_embedding


ModelType = Union[st.SentenceTransformer, LocalModel]

def __local_normalize(v: ndarray) -> ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def __local_neg_exp_mapping(start, end, num_steps, device):
    interval = end - start
    step_size = interval / num_steps
    x = torch.arange(start, end, step_size)
    c = 1 + 1/math.e ** num_steps
    y = -torch.exp(x - num_steps) + c
    y = y.to(device=device)
    return y


def __local_neg_linear_mapping(start, end, num_steps, device):
    interval = end - start
    step_size = interval / num_steps
    x = torch.arange(start, end, step_size)
    y = -1/num_steps * x + 1
    y = y.to(device=device)
    return y


def __local_text_squeeze(model: ModelType, sentences: List[str], average_t: int):
    buf = ''
    segments: List[str] = ['']
    if LOCAL_AVG_SQUEEZE == average_t:
        max_len = model.get_max_seq_length()
        for idx, s in enumerate(sentences):
            buf += s + '\n'
            tok_result = model.tokenize([buf])
            segment_len = len(tok_result['input_ids'][0])
            if segment_len >= max_len:  # overflow
                segments.append(s + '\n')
                buf = s + '\n'
            else:
                segments[-1] = buf
    else:
        segments = sentences
    return segments


def __local_average(model: ModelType, segments: List[str], weight_t: int) -> Tensor:
    embeddings = None
    for idx, s in enumerate(segments):
        embedding: Tensor = model.encode(s.strip(), convert_to_tensor=True, show_progress_bar=False)
        if embeddings is None:
            embeddings = torch.unsqueeze(embedding, 0)
        else:
            embeddings = torch.cat((embeddings, torch.unsqueeze(embedding, 0)), dim=0)

    num_steps = embeddings.size()[0]  # num rows
    if num_steps <= 1:
        return embeddings

    if weight_t == LOCAL_WEIGHT_NEG_EXP:
        weights = __local_neg_exp_mapping(0, 1, num_steps, embeddings.device)
        embeddings = embeddings * weights.view(-1, 1)
    elif weight_t == LOCAL_WEIGHT_NEG_LIN:
        weights = __local_neg_linear_mapping(0, 1, num_steps, embeddings.device)
        embeddings = embeddings * weights.view(-1, 1)

    embeddings = torch.mean(embeddings, dim=0)
    return torch.unsqueeze(embeddings, 0)


def __local_avg_trunk(model: ModelType, title_sentences: List[str], body_sentences: List[str],
                      num_seg: int = 3) -> Tensor:
    segments: List[str] = __local_text_squeeze(model, title_sentences, LOCAL_AVG_SQUEEZE)
    if body_sentences:
        segments.extend(__local_text_squeeze(model, body_sentences, LOCAL_AVG_SQUEEZE))
    if num_seg > 1:
        segments = segments[:num_seg]
    embedding: Tensor = __local_average(model, segments, LOCAL_WEIGHT_NONE)
    return embedding


def __local_extract_sentences(text: str) -> List[str]:
    marker = '# text = '
    marker_len = len(marker)
    result = []
    text_obj = obeliks.run(text, object_output=True)
    for p in text_obj:
        for s in p:
            s_meta = s['metadata']
            idx = s_meta.rfind(marker)
            if idx >= 0:
                s_data = s_meta[idx + marker_len:].strip()
                if s_data:
                    result.append(s_data)
    return result



def __local_embed(articles: List[Article], embed_field_name: str, model: ModelType,
                  cache: bool = True):
    embed = []
    for a in articles:
        if cache and a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding
                logger.debug('Loaded %s article Textonic sentence BERT embedding from cache.', a)
                continue
        embed.append(a)
    if not embed:
        return

    for a in articles:
        logging.debug("Embedding [%s] article [%s]...", embed_field_name, a)
        title_sentences = __local_extract_sentences(a.title)
        body_sentences = __local_extract_sentences(a.body)
        all_sentences = []
        all_sentences.extend(title_sentences)
        all_sentences.extend(body_sentences)
        embedding: Tensor = __local_avg_trunk(model, title_sentences, body_sentences, 3)
        # segments: List[str] = __local_text_squeeze(model, all_sentences, LOCAL_AVG_SQUEEZE)
        # embedding: Tensor = __local_average(model, segments, 1)
        a.data[embed_field_name] = __local_normalize(
            embedding.cpu()[0].numpy()
        ).tolist()
        logging.debug("Embedded [%s] article [%s].", embed_field_name, a)

    if cache:
        for a in embed:
            a.to_cache('data')  # cache article to file


def local_xlmrb_embed(articles: List[Article], embed_field_name: str, cache: bool = True):
    model_cache_dir = MODEL_CACHE_DIR
    model = LocalModel(MODEL_NAME_MAP['xlmrb'], model_cache_dir)
    return __local_embed(articles, embed_field_name, model, cache)


def local_stpara_embed(articles: List[Article], embed_field_name: str, cache: bool = True):
    model_cache_dir = MODEL_CACHE_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = st.SentenceTransformer(MODEL_NAME_MAP['st.para'], None, device, cache_folder=MODEL_CACHE_DIR)
    return __local_embed(articles, embed_field_name, model, cache)


def local_stmpnet_embed(articles: List[Article], embed_field_name: str, cache: bool = True):
    model_cache_dir = MODEL_CACHE_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = st.SentenceTransformer(MODEL_NAME_MAP['st.mpnet'], None, device, cache_folder=MODEL_CACHE_DIR)
    return __local_embed(articles, embed_field_name, model, cache)

