from typing import List, Dict
import json
import traceback
from loguru import logger

from ollama import Client

from src.utils.time.timer import log_time

# LLM_SERVER_HOST = '192.168.100.16'
LLM_SERVER_HOST = '192.168.100.10'
LLM_SERVER_PORT = 11434
client = Client(host=f'http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}')

logger.add(
    "llm_extract_output_{time}.jsonl",
    filter=lambda record: "[OUTPUT]" in record["message"],
    level="DEBUG",
    serialize=True
)
logger.add(
    "llm_extract_errors_{time}.log",
    filter=lambda record: "[COLLECT]" in record["message"],
    level="ERROR",
    serialize=True
)


class ExtractedEntityNotInText(Exception):
    pass


class InputOutputTextsNotSimilar(Exception):
    pass


def filter_extracted_entity_is_subtext(output_json: Dict) -> None:
    filterred_output_json = {}
    for k, output in output_json.items():
        text = output['text']
        append = True
        for entity in output['entities']:
            if entity[0].lower() not in text.lower():
                append = False
                error_msg = f"[COLLECT] Extracted entity '{entity[0]}' not in text '{text}'"
                (
                    logger
                    .opt(lazy=True)
                    .bind(
                        extracted_entity=entity[0],
                        text=text,
                        error_type='ExtractedEntityNotInText'
                    )
                    .error(error_msg)
                )
                output_json = {}
        if append:
            filterred_output_json[k] = output
    return filterred_output_json


def assert_similar_input_output_texts(input_texts, output_json):
    input_texts_texts = [e['text'] for e in input_texts]
    output_json_texts = [e['text'] for e in output_json.values()]
    if set(input_texts_texts) != set(output_json_texts):
        raise InputOutputTextsNotSimilar("Input texts do not match output texts")


def llm_extract(input_texts: List[Dict], system_prompt: str) -> Dict:
    """
    Params:
        input_texts (List[Dict]): contains keys `id` (str) and `text`
    """
    prompt = """
Input:
{input_texts}

Output:
"""
    prompt = prompt.format(input_texts=str(input_texts))

    response = log_time(printer=logger.debug, method_name='call_llm')(client.generate)(
        model='llama3',
        prompt=prompt,
        system=system_prompt,
        options=dict(
            temperature=0,
            top_p=1,
            # For some strange reasons setting `num_ctx` = 4096 or 1024 causes the model to return error output
            num_ctx=2048,
            num_predict=2048,
            frequency_penalty=0,
            presence_penalty=0
        ),
        format='json'
    )
    output = response['response']
    try:
        output_json = json.loads(output)
        response_metadata_fields = (
            'total_duration',
            'load_duration',
            'prompt_eval_count',
            'prompt_eval_duration',
            'eval_count',
            'eval_duration',
        )
        response_metadata = {
            k: v for k, v in response.items() if k in response_metadata_fields
        }
        response_metadata = {
            **response_metadata,
            "num_chars_prompt": len(prompt),
            "num_chars_system_prompt": len(system_prompt),
        }
        (
            logger
            .opt(lazy=True)
            .bind(
                input_texts=input_texts,
                llm_extracted=json.loads(output),
                response_metdata=response_metadata,
            )
            .debug('[OUTPUT] LLM Extracted successfully')
        )
        assert_similar_input_output_texts(input_texts, output_json)
        output_json = filter_extracted_entity_is_subtext(output_json)
    except Exception as e:
        error_msg = f"[COLLECT] {traceback.format_exc()}"
        (
            logger
            .opt(lazy=True)
            .bind(
                input_texts=input_texts,
                llm_extracted=output,
                error_type=e.__class__.__name__
            )
            .error(error_msg)
        )
        output_json = {}
    return output_json


def llm_extract_modelfile(input_texts: List[Dict]) -> Dict:
    prompt = """
Input:
{input_texts}

Output:
"""
    prompt = prompt.format(input_texts=str(input_texts))

    response = log_time(printer=logger.debug, method_name='call_llm')(client.generate)(
        model='ner_aspect_review',
        prompt=prompt,
        format='json'
    )
    output = response['response']
    try:
        output_json = json.loads(output)
        (
            logger
            .opt(lazy=True)
            .bind(input_texts=input_texts, llm_extracted=output)
            .debug('[OUTPUT] LLM Extracted successfully')
        )
        input_texts_texts = [e['text'] for e in input_texts]
        output_json_texts = [e['text'] for e in output_json.values()]
        assert set(input_texts_texts) == set(output_json_texts), "Input texts do not match output texts"
        assert_extracted_entity_is_subtext(output_json)
    except json.decoder.JSONDecodeError:
        error_msg = f"[COLLECT] {traceback.format_exc()}"
        (
            logger
            .opt(lazy=True)
            .bind(
                input_texts=input_texts,
                llm_extracted=output,
                error_type='JSONOutputFormatError'
            )
            .error(error_msg)
        )
        output_json = {}
    return output_json
