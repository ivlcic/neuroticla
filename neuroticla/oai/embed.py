import logging
import openai

from typing import List

from ..esdl.article import Article
from .tokenize import truncate_text_tokens
from .constants import EMBEDDING_ENCODING, EMBEDDING_CTX_LENGTH

logger = logging.getLogger('oai.embed')


def openai_embed(articles: List[Article], embed_field_name: str, cache: bool = True):
    for a in articles:
        if a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article OpenAI embedding from cache.', a)
                continue
        logger.debug('Loading %s article OpenAI embedding ...', a)
        tokens = truncate_text_tokens(
            a.title + ' ' + a.body,
            EMBEDDING_ENCODING,
            EMBEDDING_CTX_LENGTH
        )
        embedding = openai.Embedding.create(  # call OpenAI
            input=tokens, model="text-embedding-ada-002"
        )
        logger.info('Loaded %s article OpenAI embedding.', a)
        a.data[embed_field_name] = embedding["data"][0]["embedding"]  # extract vector from response
        if cache:
            a.to_cache('data')  # cache article to file
