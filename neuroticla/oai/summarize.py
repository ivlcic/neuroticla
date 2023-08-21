import logging
import math
import openai
import tiktoken

from typing import List
from ..esdl.article import Article
from .tokenize import create_chunks
from .constants import DEFAULT_MODEL, DEFAULT_S_REPLACE, MODEL_TOKENS, MODEL_ABBREV

logger = logging.getLogger('oai.summarize')


def __call_openai_completion(text, prompt_template, search_replace, max_model_len, tokenizer, model, max_tokens):
    dec = 0
    if prompt_template:
        dec = len(tokenizer.encode(prompt_template))
    else:
        dec = len(tokenizer.encode("\n\nTl;dr"))

    max_model_len = max_model_len - dec - max_tokens
    chunks = create_chunks(text, max_model_len, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    summary = []
    max_tokens = math.ceil(max_tokens / len(text_chunks))
    for chunk in text_chunks:
        if prompt_template:
            chunk = prompt_template.replace(search_replace, chunk)
        else:
            chunk += "\n\nTl;dr"

        response = openai.Completion.create(
            model=model,
            prompt=chunk,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=1
        )
        resp: str = response["choices"][0]["text"]
        if resp.startswith(':'):
            resp = resp[1:]
        summary.append(resp.strip())
    return ''.join(summary)


def summarize_article(articles: List[Article], max_tokens=512, summary_field_name: str = None,
                      prompt_template: str = None, model=DEFAULT_MODEL,
                      search_replace: str = DEFAULT_S_REPLACE):
    if model not in MODEL_TOKENS:
        raise RuntimeError('Invalid model specified!')
    max_model_len = MODEL_TOKENS[model]
    model_abbrev = MODEL_ABBREV[model]
    if not summary_field_name:
        summary_field_name = 'summary-' + model_abbrev

    #max_model_len = 512
    tokenizer = tiktoken.encoding_for_model(model)
    logger.debug("Summarizing [%s] articles with model [%s] max len [%s].",
                 model, max_model_len, len(articles))
    for a in articles:
        if a.from_cache('data'):
            if summary_field_name in a.data:
                continue

        text = a.title + '\n\n' + a.body
        if len(text) < 255:
            a.data[summary_field_name] = ''
            continue
        logger.debug("Summarizing %s ...", a)
        summary = __call_openai_completion(text, prompt_template, search_replace,
                                           max_model_len, tokenizer, model, max_tokens)
        logger.info("Summarized %s with summary [%s:%s][%s]",
                    a, len(tokenizer.encode(summary)), len(summary), summary)
        if summary_field_name not in a.data:
            a.data[summary_field_name] = ''
        a.data[summary_field_name] += summary

        a.to_cache('data')  # cache article to file
    logger.info("Summarized [%s] articles.", len(articles))
