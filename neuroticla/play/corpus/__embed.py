import logging
import regex
import openai
import torch.nn.functional as functional

from torch import Tensor

from .__utils import Params
from ...ner.prep.tokens import get_tokenizer
from ...esdl.article import Article
from ...oai.constants import EMBEDDING_ENCODING, EMBEDDING_CTX_LENGTH
from ...oai.tokenize import truncate_text_tokens

logger = logging.getLogger('play.cluster.embed')

__SOURCE_PATT = regex.compile(r'^([\p{Nd}\p{Lu}\s]+([,\s*]))*(\s*(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})([,\s*])*(\s*\p{Lu}{2,}[\p{Nd}\p{Lu}\s]+([,\s*])*(\s*\d{1,2}:\d{0,2})?)?)?\s*$')
__ACTOR_PATT = regex.compile(r'^[\p{Nd}\p{Lu}\s]+(\([\s\p{L}\p{Nd}\p{P}]+\))?\s*$')


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _filter_body(sentence_idx, sentence):
    if sentence_idx == 0:  # first sentence of a body section
        match = regex.match(__SOURCE_PATT, sentence)
    else:  # other sentences of a body section
        match = regex.match(__ACTOR_PATT, sentence)
    if match is None:
        return sentence
    else:
        return ''


def _e5_embed(params: Params, text):
    batch_dict = params.tokenizer(
        ['passage: ' + text], max_length=512,
        padding=True, truncation=True, return_tensors='pt'
    )
    outputs = params.model(**batch_dict)
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()[0]


def _e5_token_compute(params: Params, text):
    tokens = params.tokenizer(
        ['passage: ' + text], return_tensors='pt'
    )
    return len(tokens.encodings[0].tokens)


def _oai_embed(text):
    tokens = truncate_text_tokens(
        text,
        EMBEDDING_ENCODING,
        EMBEDDING_CTX_LENGTH
    )
    embedding = openai.embeddings.create(  # call OpenAI
        input=tokens, model="text-embedding-ada-002", timeout=10
    )
    return embedding.data[0].embedding  # extract vector from response


def _oai_token_compute(text):
    tokens = truncate_text_tokens(
        text,
        EMBEDDING_ENCODING,
        EMBEDDING_CTX_LENGTH
    )
    return len(tokens)


def _embed(params: Params, article: Article):
    params.lang = article.language.split('-', 1)[0]
    if params.lang == 'bs':  # since we don't have any tokenizer for Bosnian
        params.lang = 'hr'
    if params.lang == 'sq':  # since we don't have any tokenizer for Albanian
        params.lang = 'en'

    tokenizer = get_tokenizer(params)

    body = ''
    filtered = False
    mt = article.data['media']['type']['name']
    if mt == 'radio' or mt == 'tv':
        for line_idx, line in enumerate(article.body.split('\n')):
            tmp = _filter_body(line_idx, line)
            if tmp:
                body += '\n' if len(body) > 0 else ''
                body += tmp
            if not tmp and line:
                filtered = True
    if not body:
        body = article.body

    tdoc = tokenizer(article.title)
    bdoc = tokenizer(body)
    blen = len(body)

    bwt = 0
    for sentence in bdoc.sentences:
        bwt += len(sentence.tokens)
    twt = 0
    for sentence in tdoc.sentences:
        twt += len(sentence.tokens)

    b_spt = _e5_token_compute(params, body)
    t_spt = _e5_token_compute(params, article.title)

    b_oait = _oai_token_compute(body)
    t_oait = _oai_token_compute(article.title)

    article.data['body'] = {
        'text': article.body,
        'stats': {
            'chr': blen,
            'sent': len(bdoc.sentences),
            'w_t': bwt,
            'sp_t': b_spt,
            'oai_t': b_oait,
            'filter': True if filtered else False
        }
    }
    article.data['title'] = {
        'text': article.title,
        'stats': {
            'chr': len(article.title),
            'sent': len(tdoc.sentences),
            'w_t': twt,
            'sp_t': t_spt,
            'oai_t': t_oait,
        }
    }

    if not params.skipEmbedding:
        article.data['embed_e5'] = _e5_embed(params, article.title + ' ' + body)
        article.data['embed_oai'] = _oai_embed(article.title + ' ' + body)

    article.data['stats'] = {
        'chr': article.data['title']['stats']['chr'] + article.data['body']['stats']['chr'],
        'sent': article.data['title']['stats']['sent'] + article.data['body']['stats']['sent'],
        'w_t': article.data['title']['stats']['w_t'] + article.data['body']['stats']['w_t'],
        'sp_t': article.data['title']['stats']['sp_t'] + article.data['body']['stats']['sp_t'],
        'oai_t': article.data['title']['stats']['oai_t'] + article.data['body']['stats']['oai_t']
    }

    if not params.skipEmbedding:
        logger.info("Done embedding [%s]", article)


