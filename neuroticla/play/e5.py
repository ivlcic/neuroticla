import logging
import torch.nn.functional as functional

from typing import List, Union
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from ..esdl.article import Article

logger = logging.getLogger('e5.embed')


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def e5_embed(articles: List[Article], embed_field_name: str, cache: Union[str, None] = 'data'):
    if embed_field_name.startswith('efed'):
        model_name = 'efederici/e5-base-multilingual-4096'
        max_len = 4096
    else:
        model_name = 'intfloat/multilingual-e5-base'
        max_len = 512
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    for a in articles:
        if cache is not None and a.from_cache(cache):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article E5 embedding from cache.', a)
                continue
        logger.debug('Loading %s article E5 embedding ...', a)
        batch_dict = tokenizer(
            ['passage: ' + a.title + ' ' + a.body], max_length=max_len,
            padding=True, truncation=True, return_tensors='pt'
        )
        outputs = model(**batch_dict)
        embeddings = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = functional.normalize(embeddings, p=2, dim=1)
        logger.info('Loaded %s article E5 embedding.', a)
        a.data[embed_field_name] = embeddings.tolist()[0]  # extract vector from response
        if cache:
            a.to_cache(cache)  # cache article to file
