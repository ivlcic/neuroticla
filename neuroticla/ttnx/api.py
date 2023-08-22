import logging
import os
import json
import requests

from typing import Any, Dict, List

from .constants import TTNX_API_KEY

logger = logging.getLogger('ttnx.api')


def call_textonic(url_path: str, json_object: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = 'https://textonic.io' + url_path
    # url = 'http://localhost:8080' + url_path
    api_key = os.environ[TTNX_API_KEY]
    query = json.dumps(json_object)
    result = []
    logger.debug('Invoking Textonic [%s]...', url)
    try:
        # make HTTP verb parameter case-insensitive by converting to lower()
        resp = requests.post(url,
                             headers={
                                 'Content-Type': 'application/json',
                                 'Accept': 'application/vnd.dropchop.result+json',
                                 'Accept-Encoding': 'gzip,deflate',
                                 'Authorization': 'Bearer ' + api_key
                             },
                             data=query)
    except Exception as error:
        logger.error('Textonic request [%s] error [%s]:', query, error)
        return result
    logger.info('Invoked Textonic [%s]', url)
    try:
        resp_obj: Dict[str, Any] = json.loads(resp.text)
        if resp_obj['status']['code'] == 'success':
            return resp_obj
        else:
            logger.error(
                "Textonic error [%s][%s]",
                resp_obj['status']['message']['code'],
                resp_obj['status']['message']['details']
            )
            raise RuntimeError('Textonic error!')
    except:
        logger.error('Textonic parse error [%s]:', resp.text)
    logger.info('Parsed Textonic [%s] articles.', len(result))
    return result
