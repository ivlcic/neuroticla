import os
import logging
import json
import regex
import openai
import torch.nn.functional as functional

from torch import Tensor

from ...ner.prep.tokens import get_tokenizer
from ...esdl.article import Article
from ...oai.constants import EMBEDDING_ENCODING, EMBEDDING_CTX_LENGTH
from ...oai.tokenize import truncate_text_tokens

logger = logging.getLogger('play.cluster.embed')

__SOURCE_PATT = regex.compile(r'^[\p{Nd}\p{Lu}\s]+([,\s*])*(\s*(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})([,\s*])*(\s*\p{Lu}{2,}[\p{Nd}\p{Lu}\s]+([,\s*])*(\s*\d{1,2}:\d{0,2})?)?)?\s*$')
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


def _e5_embed(arg, text):
    batch_dict = arg.tokenizer(
        ['passage: ' + text], max_length=512,
        padding=True, truncation=True, return_tensors='pt'
    )
    outputs = arg.model(**batch_dict)
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()[0]


def _e5_token_compute(arg, text):
    tokens = arg.tokenizer(
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


def _embed(arg, article: Article):
    arg.lang = article.language.split('-', 1)[0]
    if arg.lang == 'bs':  # since we don't have any tokenizer for Bosnian
        arg.lang = 'hr'
    if arg.lang == 'sq':  # since we don't have any tokenizer for Albanian
        arg.lang = 'en'

    tokenizer = get_tokenizer(arg)

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

    b_spt = _e5_token_compute(arg, body)
    t_spt = _e5_token_compute(arg, article.title)

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

    article.data['embed_e5'] = _e5_embed(arg, article.title + ' ' + body)
    article.data['embed_oai'] = _oai_embed(article.title + ' ' + body)

    article.data['stats'] = {
        'chr': article.data['title']['stats']['chr'] + article.data['body']['stats']['chr'],
        'sent': article.data['title']['stats']['sent'] + article.data['body']['stats']['sent'],
        'w_t': article.data['title']['stats']['w_t'] + article.data['body']['stats']['w_t'],
        'sp_t': article.data['title']['stats']['sp_t'] + article.data['body']['stats']['sp_t'],
        'oai_t': article.data['title']['stats']['oai_t'] + article.data['body']['stats']['oai_t']
    }


def _filter_write(arg, article: Article, data_path: str):
    article.data.pop('advertValue', None)
    article.data.pop('mediaReach', None)
    article.data.pop('translations', None)

    media = article.data.get('media')
    media.pop('publisher', None)
    media.pop('advertValue', None)
    media.pop('circulation', None)
    media.pop('providerId', None)
    media.pop('descriptor', None)
    media.pop('class', None)
    media.pop('language', None)
    tags = media.pop('tags', [])
    tags[:] = [tup for tup in tags if tup.get('class', '') == 'org.dropchop.jop.beans.tags.MediaType']
    media['type'] = {
        'name': tags[0]['name'],
        'uuid': tags[0]['uuid']
    }

    rubric = article.data.get('rubric')
    rubric.pop('advertValue', None)
    rubric.pop('providerId', None)
    rubric.pop('descriptor', None)
    rubric.pop('class', None)

    country = article.data.get('country')
    country.pop('tags', None)
    country.pop('descriptor', None)
    country.pop('class', None)

    for k, v in article.data.get('translations', {}).items():
        v.pop('bodyPages')
        v.pop('bodyOctetLen')
        v.pop('bodyCalculatedPages')
        v.pop('bodyMd5')
        v.pop('bodyBillingPages')
        v.pop('class')
        v.pop('uuid')

    tags = article.data.pop('tags', [])
    tags[:] = [tup for tup in tags if tup.get('class', '') != 'org.dropchop.jop.beans.tags.Genre']
    for i, t in enumerate(tags):
        t.pop('descriptor', None)
        t.pop('name', None)
        t.pop('_OOP', None)
        if t.pop('class', '') == 'org.dropchop.jop.beans.tags.CustomerTopic':
            t['type'] = 'topic'
        if t.pop('class', '') == 'org.dropchop.jop.beans.tags.CustomerTopicGroup':
            t['type'] = 'group'

    _embed(arg, article)
    logger.info("Done filtering and embedding [%s]", article)

    article.data['tags'] = tags
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, article.uuid + '.json'), 'w', encoding='utf8') as json_file:
        json.dump(article.data, json_file, indent='  ', ensure_ascii=False)
