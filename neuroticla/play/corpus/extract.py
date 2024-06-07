import csv
import json
import logging
import os
import openai

from typing import List, Dict, Any


from .__embed import _filter_body
from .__utils import load_range, State, Params
from ...esdl.article import Article

logger = logging.getLogger('play.corpus.dump')

system_prompt_1 = """
You will be given comma-separated terms and a news article from which you must construct at most twenty question-and-answer pairs. 
You must use all combinations of terms for questions. 
The question must relate to the content of a news article passage, and the answer must be a verbatim text passage from an article over 250 characters. 
You MUST use the article language for questions and answers.
You will respond with a JSON object containing a "pairs" property which is a list of question-and-answer text pairs from an article. 
Each pair has properties "q" for the question and "a" for an answer.
"""

def _qagen(prompt):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_organization = os.getenv('OPENAI_ORG_ID')
    openai_model = os.getenv('OPENAI_MODEL')

    client = openai.OpenAI(
        api_key=openai_api_key,
        organization=openai_organization
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_1},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
    except openai.APIConnectionError as e:
        # Handle connection error here
        msg = f"Failed to connect to OpenAI API: {e}. Please try again later."
        logger.error(msg)
        return {"error": msg}
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        msg = f"OpenAI API request exceeded rate limit: {e}. Please try again later."
        logger.error(msg)
        return {"error": msg}
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        msg = f"OpenAI API returned an API Error: {e}. Please try again later."
        logger.error(msg)
        return {"error": msg}
    try:
        # Extract the JSON string from the response
        chat_message_content = completion.choices[0].message.content
        data = json.loads(chat_message_content)
        return data
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Failed to decode JSON response from OpenAI: {str(e)}")
        msg = "Received an invalid response. Please try again with a different query."
        return {"error": msg}


def _extract_substring(text, start_index, end_index):
    if start_index < 0:
        start_index = 0  # Clamp to the start of the string
    if end_index > len(text):
        end_index = len(text)  # Clamp to the end of the string
    if start_index >= end_index:
        return text         # Return the whole text if indices are reversed or equal

    return text[start_index:end_index]


def _write_spans(kwes: Dict[str, Any], saved, matches, text, f_name):
    t_names = set()
    k_matches = set()
    for t in saved['tags']:
        t_uuid = t['uuid']
        t[f_name] = []
        if 'name' in t and t['name']:
            t_names.add(t['name'])
        for s in matches['spans']:
            for k_uuid in s['kwe']:
                if t_uuid == matches['kwe'][k_uuid]['topic']['uuid']:
                    k_part = k_uuid.split('-')[0]
                    if k_part not in kwes:
                        kwes[k_part] = {
                            'uuid': k_uuid,
                            'expr': matches['kwe'][k_uuid]['expr'],
                            'parent': t_uuid
                        }
                    m = _extract_substring(text, s['start'], s['end'])
                    if m:
                        k_matches.add(m)
                    t[f_name].append(
                        {
                            's': s['start'],
                            'e': s['end'],
                            'm': m,
                            'kwe': k_part
                        }
                    )
    return t_names.union(k_matches)


def load_map_file(file_name: str, cols: List[str]) -> Dict[str, Dict[str, Any]]:
    d = {}
    if not os.path.exists(file_name):
        return d
    with open(file_name, encoding='utf-8') as d_file:
        try:
            d_reader = csv.reader(d_file)
            for row_idx, row in enumerate(d_reader):
                if row_idx == 0:
                    continue
                key = row[0]
                d[key] = {}
                for idx, c in enumerate(cols):
                    d[key][c] = row[idx + 1]
        except Exception as e:
            logger.error("Unable to load CSV kwe map file [%s].", file_name, e)
            exit(1)
    return d


def write_map_file(d: Dict[str, Dict[str, Any]], file_name: str, cols: List[str]) -> None:
    with open(file_name, 'w', encoding='utf8') as kwe_file:
        writer = csv.writer(kwe_file)
        row = ['id']
        for c in cols:
            row.append(c)
        writer.writerow(row)
        for k, v in d.items():
            row = [k]
            for c in cols:
                row.append(v[c])
            writer.writerow(row)


def extract_data(arg) -> int:
    token = os.environ.get('KMAP_TOKEN')
    if not token:
        logger.error('Missing KMAP_TOKEN environment variable')
        return 1

    save_dir = os.path.join(arg.result_dir, 'corpus')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    params = Params(arg.start_date, arg.end_date, None, arg.result_dir)

    kwe_file_name = os.path.join(save_dir, 'map_kwes.csv')
    kwe_cols = ['uuid', 'expr', 'parent']
    kwes = load_map_file(kwe_file_name, kwe_cols)

    topic_file_name = os.path.join(save_dir, 'map_topics.csv')
    topic_cols = ['uuid', 'name', 'count', 'parent']
    topics = load_map_file(topic_file_name, topic_cols)

    media_file_name = os.path.join(save_dir, 'map_media.csv')
    media_cols = ['uuid', 'name', 'country', 'count', 'type', 'reach', 'url']
    medias = load_map_file(media_file_name, media_cols)

    article_file_name = os.path.join(save_dir, f'map_articles_{params.start.year}_{params.start.month:02d}.csv')
    article_cols = [
        'uuid', 'created', 'published', 'country', 'lang', 'scr', 'm_id', 'rel_path', 'filtered', 'url', 'sent',
        'words', 'sp_tokens', 'n_tags', 'tags'
    ]
    articles = load_map_file(article_file_name, article_cols)

    data_file_name = f'data_{params.start.year}_{params.start.month:02d}.jsonl'
    data = []

    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        if not os.path.exists(s.file):
            logger.warning("Unable to find article file %s", s.file)
            return 0
        if len(saved) == 1 and 'ver' in saved:
            logger.warning("Corrupted article file %s", s.file)
            os.remove(s.file)
            return 0

        if 'tags' not in saved or len(saved['tags']) == 0:
            logger.warning("Article %s has no tags.", s.file)
            os.remove(s.file)
            return 0

        pdate = saved.pop('published')
        country = saved.pop('country')
        media = saved.pop('media')
        m_part = media['uuid'].split('-')[0]
        if m_part not in medias:
            medias[m_part] = {
                'uuid': media['uuid'],
                'name': media['name'],
                'country': country['name'],
                'type': media['type']['name'],
                'reach': media['mediaReach'] if 'mediaReach' in media else 0,
                'count': 1,
                'url':  media['url'] if 'url' in media else None,
            }
        else:
            medias[m_part]['count'] = int(medias[m_part]['count']) + 1
        saved.pop('rubric', None)
        saved.pop('embed_e5', None)
        saved.pop('embed_oai', None)
        saved.pop('ver', None)
        url = saved.pop('url', None)
        cdate = saved.pop('created')
        saved['date'] = cdate
        uuid = saved.pop('uuid')
        saved['id'] = uuid.split('-')[0]
        saved['country'] = country['name']
        saved['lang'] = saved.pop('language').split('-')[0]
        saved['m_id'] = m_part

        title = saved.pop('title')
        saved['title'] = title
        is_cyrillic = False
        if title['text']:
            is_cyrillic = any(0x0400 <= ord(char) <= 0x04ff for char in title['text'])
            saved['title']['scr'] = 'latn' if not is_cyrillic else 'cyrl'
        if saved['body']['text']:
            is_cyrillic = any(0x0400 <= ord(char) <= 0x04ff for char in saved['body']['text'])
            saved['body']['scr'] = 'latn' if not is_cyrillic else 'cyrl'

        stats = saved['title'].pop('stats', None)
        saved['title']['stat'] = [stats['sent'], stats['w_t'], stats['sp_t']]

        stats = saved['body'].pop('stats', None)
        saved['body']['stat'] = [stats['sent'], stats['w_t'], stats['sp_t']]

        matches = saved['title'].pop('matches')
        title_kwords = _write_spans(kwes, saved, matches, saved['title']['text'], 'ts')
        matches = saved['body'].pop('matches')
        body_kwords = _write_spans(kwes, saved, matches, saved['body']['text'], 'bs')
        body_kwords.union(title_kwords)

        # prompt = f"{body_kwords}\n\n{saved['title']['text']}\n\n{saved['body']['text']}"
        # qa_data = _qagen(prompt)

        tags = saved.pop('tags')
        t_uuids = set()
        for t in tags:
            t_uuid = t.pop('uuid')
            t.pop('type')
            name = t.pop('name', None)
            t_part = t_uuid.split('-')[0]
            t_uuids.add(t_part)
            t['id'] = t_part
            t['bs'] = t.pop('bs')
            t['ts'] = t.pop('ts')
            parent = t.pop('parent', None)
            if t_part not in topics:
                topics[t_part] = {
                    'uuid': t_uuid,
                    'name': name,
                    'count': 1,
                    'parent': parent['uuid']
                }
            else:
                topics[t_part]['count'] = int(topics[t_part]['count']) + 1
            if parent is not None and 'uuid' in parent:
                t['parent'] = parent['uuid'].split('-')[0]
        saved['tags'] = tags
        saved['kw'] = list(body_kwords)

        b_dict = saved.pop('body')
        body = ''
        mt = media['type']['name']
        filtered = False
        if mt == 'radio' or mt == 'tv':
            for line_idx, line in enumerate(b_dict['text'].split('\n')):
                tmp = _filter_body(line_idx, line)
                if tmp:
                    body += '\n' if len(body) > 0 else ''
                    body += tmp
                if not tmp and line:
                    body += '\n' if len(body) > 0 else ''
                    body += " " * len(line)
                    filtered = True

        if body:
            b_dict['text'] = body

        saved['body'] = b_dict
        stats = saved.pop('stats')
        articles[saved['id']] = {
            'uuid': uuid,
            'created': cdate,
            'published': pdate,
            'country': country['name'],
            'lang': saved['lang'],
            'scr': 'latn' if not is_cyrillic else 'cyrl',
            'm_id': saved['m_id'],
            'rel_path': s.relDir,
            'filtered': filtered,
            #  'file': data_file_name,
            'url': url,
            'sent': stats['sent'],
            'words': stats['w_t'],
            'sp_tokens': stats['sp_t'],
            'n_tags': len(t_uuids),
            'tags': list(t_uuids)
        }

        data.append(saved)
        return 1

    state = load_range(params, callback)

    file_name = os.path.join(save_dir, data_file_name)
    with open(file_name, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent='  ', ensure_ascii=False)
    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    write_map_file(kwes, kwe_file_name, kwe_cols)
    write_map_file(topics, topic_file_name, topic_cols)
    write_map_file(medias, media_file_name, media_cols)
    write_map_file(articles, article_file_name, article_cols)

    return 0


def extract_embeddings(arg) -> int:
    token = os.environ.get('KMAP_TOKEN')
    if not token:
        logger.error('Missing KMAP_TOKEN environment variable')
        return 1

    save_dir = os.path.join(arg.result_dir, 'corpus')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    params = Params(arg.start_date, arg.end_date, None, arg.result_dir)

    kwe_file_name = os.path.join(save_dir, 'map_kwes.csv')
    kwe_cols = ['uuid', 'expr', 'parent']
    kwes = load_map_file(kwe_file_name, kwe_cols)

    topic_file_name = os.path.join(save_dir, 'map_topics.csv')
    topic_cols = ['uuid', 'name', 'count', 'parent']
    topics = load_map_file(topic_file_name, topic_cols)

    media_file_name = os.path.join(save_dir, 'map_media.csv')
    media_cols = ['uuid', 'name', 'country', 'count', 'type', 'reach', 'url']
    medias = load_map_file(media_file_name, media_cols)

    article_file_name = os.path.join(save_dir, f'map_articles_{params.start.year}_{params.start.month:02d}.csv')
    article_cols = [
        'uuid', 'created', 'published', 'country', 'lang', 'scr', 'm_id', 'rel_path', 'filtered', 'url', 'sent',
        'words', 'sp_tokens', 'n_tags', 'tags'
    ]
    articles = load_map_file(article_file_name, article_cols)

    data_file_name = f'data_{params.start.year}_{params.start.month:02d}.jsonl'
    data = []

    def callback(s: State, saved: Dict[str, Any], article: Article) -> int:
        if not os.path.exists(s.file):
            logger.warning("Unable to find article file %s", s.file)
            return 0
        if len(saved) == 1 and 'ver' in saved:
            logger.warning("Corrupted article file %s", s.file)
            os.remove(s.file)
            return 0

        if 'tags' not in saved or len(saved['tags']) == 0:
            logger.warning("Article %s has no tags.", s.file)
            os.remove(s.file)
            return 0

        pdate = saved.pop('published')
        country = saved.pop('country')
        media = saved.pop('media')
        m_part = media['uuid'].split('-')[0]
        if m_part not in medias:
            medias[m_part] = {
                'uuid': media['uuid'],
                'name': media['name'],
                'country': country['name'],
                'type': media['type']['name'],
                'reach': media['mediaReach'] if 'mediaReach' in media else 0,
                'count': 1,
                'url':  media['url'] if 'url' in media else None,
            }
        else:
            medias[m_part]['count'] = int(medias[m_part]['count']) + 1
        saved.pop('rubric', None)
        saved.pop('embed_e5', None)
        saved.pop('embed_oai', None)
        saved.pop('ver', None)
        url = saved.pop('url', None)
        cdate = saved.pop('created')
        saved['date'] = cdate
        uuid = saved.pop('uuid')
        saved['id'] = uuid.split('-')[0]
        saved['country'] = country['name']
        saved['lang'] = saved.pop('language').split('-')[0]
        saved['m_id'] = m_part

        title = saved.pop('title')
        saved['title'] = title
        is_cyrillic = False
        if title['text']:
            is_cyrillic = any(0x0400 <= ord(char) <= 0x04ff for char in title['text'])
            saved['title']['scr'] = 'latn' if not is_cyrillic else 'cyrl'
        if saved['body']['text']:
            is_cyrillic = any(0x0400 <= ord(char) <= 0x04ff for char in saved['body']['text'])
            saved['body']['scr'] = 'latn' if not is_cyrillic else 'cyrl'

        stats = saved['title'].pop('stats', None)
        saved['title']['stat'] = [stats['sent'], stats['w_t'], stats['sp_t']]

        stats = saved['body'].pop('stats', None)
        saved['body']['stat'] = [stats['sent'], stats['w_t'], stats['sp_t']]

        matches = saved['title'].pop('matches')
        title_kwords = _write_spans(kwes, saved, matches, saved['title']['text'], 'ts')
        matches = saved['body'].pop('matches')
        body_kwords = _write_spans(kwes, saved, matches, saved['body']['text'], 'bs')
        body_kwords.union(title_kwords)

        # prompt = f"{body_kwords}\n\n{saved['title']['text']}\n\n{saved['body']['text']}"
        # qa_data = _qagen(prompt)

        tags = saved.pop('tags')
        t_uuids = set()
        for t in tags:
            t_uuid = t.pop('uuid')
            t.pop('type')
            name = t.pop('name', None)
            t_part = t_uuid.split('-')[0]
            t_uuids.add(t_part)
            t['id'] = t_part
            t['bs'] = t.pop('bs')
            t['ts'] = t.pop('ts')
            parent = t.pop('parent', None)
            if t_part not in topics:
                topics[t_part] = {
                    'uuid': t_uuid,
                    'name': name,
                    'count': 1,
                    'parent': parent['uuid']
                }
            else:
                topics[t_part]['count'] = int(topics[t_part]['count']) + 1
            if parent is not None and 'uuid' in parent:
                t['parent'] = parent['uuid'].split('-')[0]
        saved['tags'] = tags
        saved['kw'] = list(body_kwords)

        b_dict = saved.pop('body')
        body = ''
        mt = media['type']['name']
        filtered = False
        if mt == 'radio' or mt == 'tv':
            for line_idx, line in enumerate(b_dict['text'].split('\n')):
                tmp = _filter_body(line_idx, line)
                if tmp:
                    body += '\n' if len(body) > 0 else ''
                    body += tmp
                if not tmp and line:
                    body += '\n' if len(body) > 0 else ''
                    body += " " * len(line)
                    filtered = True

        if body:
            b_dict['text'] = body

        saved['body'] = b_dict
        stats = saved.pop('stats')
        articles[saved['id']] = {
            'uuid': uuid,
            'created': cdate,
            'published': pdate,
            'country': country['name'],
            'lang': saved['lang'],
            'scr': 'latn' if not is_cyrillic else 'cyrl',
            'm_id': saved['m_id'],
            'rel_path': s.relDir,
            'filtered': filtered,
            #  'file': data_file_name,
            'url': url,
            'sent': stats['sent'],
            'words': stats['w_t'],
            'sp_tokens': stats['sp_t'],
            'n_tags': len(t_uuids),
            'tags': list(t_uuids)
        }

        data.append(saved)
        return 1

    state = load_range(params, callback)

    file_name = os.path.join(save_dir, data_file_name)
    with open(file_name, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent='  ', ensure_ascii=False)
    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )
    write_map_file(kwes, kwe_file_name, kwe_cols)
    write_map_file(topics, topic_file_name, topic_cols)
    write_map_file(medias, media_file_name, media_cols)
    write_map_file(articles, article_file_name, article_cols)

    return 0
