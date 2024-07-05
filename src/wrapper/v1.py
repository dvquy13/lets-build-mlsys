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
    "llm_extract_output_{time}.log",
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

def llm_extract(input_texts: List[Dict], system_prompt: str) -> Dict:
    """
    Params:
        input_texts (List[Dict]): contains keys `id` (str) and `text`
    """
    prompt = """
    Input:
    {input_texts}
    """

    response = log_time(printer=logger.debug, method_name='call_llm')(client.chat)(
        model='llama3',
        messages=[
            {
              "role": "system",
              "content": system_prompt
            },
            {
              "role": "user",
              "content": prompt.format(input_texts=str(input_texts))
            }
        ],
        options=dict(
            temperature=0,
            num_predict=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        ),
        format='json'
    )
    output = response['message']['content']
    try:
        output_json = json.loads(output)
        logger.opt(lazy=True).bind(llm_extracted=output).debug('[OUTPUT] LLM Extracted successfully')
    except json.decoder.JSONDecodeError:
        error_msg = f"[COLLECT] {traceback.format_exc()}"
        logger.opt(lazy=True).bind(text=output, error_type='JSONOutputFormatError').error(error_msg)
        output_json = {}
    return output_json
