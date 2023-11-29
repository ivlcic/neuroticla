import logging
import os.path

import torch.nn.functional as functional

from typing import List, Union
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from ..esdl.article import Article

logger = logging.getLogger('e5.embed')


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _e5_embed(tokenizer, model, text, max_len):
    batch_dict = tokenizer(
        ['passage: ' + text], max_length=max_len,
        padding=True, truncation=True, return_tensors='pt'
    )
    outputs = model(**batch_dict)
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def e5_embed_text(tmp_dir: str, text):
    max_len = 512
    model_name = 'intfloat/multilingual-e5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=os.path.join(tmp_dir, model_name))
    embeddings = _e5_embed(tokenizer, model, text, max_len)
    return embeddings.tolist()[0]


def e5_embed(articles: List[Article], embed_field_name: str, tmp_dir: str, fields: str = None,
             large_model: bool = False, cache: Union[str, None] = None):
    if embed_field_name.startswith('efed'):
        model_name = 'efederici/e5-base-multilingual-4096'
        max_len = 4096
    else:
        model_name = 'intfloat/multilingual-e5-base'
        max_len = 512

    if large_model:
        model_name = 'intfloat/multilingual-e5-large'
        max_len = 512

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=os.path.join(tmp_dir, model_name))

    for a in articles:
        if cache is not None and a.from_cache(cache):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article E5 embedding from cache.', a)
                continue
        logger.debug('Loading %s article E5 embedding ...', a)
        text = a.title + ' ' + a.body
        if fields == 'b':
            text = a.body
            if not text or not text.strip():
                text = a.title
        embeddings = _e5_embed(tokenizer, model, text, max_len)
        logger.info('Loaded %s article E5 embedding.', a)
        a.data[embed_field_name] = embeddings.tolist()[0]  # extract vector from response
        if cache:
            a.to_cache(cache)  # cache article to file
