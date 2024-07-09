from loguru import logger

from src.utils.id.idfy import deterministic_hash
from src.utils.time.timer import log_time


@log_time(printer=logger.info)
def convert_from_llm_output_to_spacy(llm_output):
    """
    LLM output format:
        [
            {
                "1": {
                    "text": "The service is not consistently excellent -- just decent.",
                    "entities": [
                        [
                            "service is not consistently excellent",
                            "SERVICE",
                            0.4,
                            -0.3
                        ],
                        [
                            "just decent",
                            "SERVICE",
                            0.5,
                            0.2
                        ]
                    ]
                }
            }
        ]

    SpaCy format:
        [
            {
                "id": "339593553392192228390882582566160737164",
                "text": "The place was nice and calm.",
                "label": [[4, 27, "AMBIENCE"]],
                "Comments": []
            }
        ]
    """
    spacy_data = []
    for item in llm_output:
        text = item["text"]
        entities = item["entities"]
        labels = []
        for entity in entities:
            if not isinstance(entity, list) or len(entity) != 4:
                logger.error(
                    "\n".join(
                        [
                            f"Entity should be a list like ['text', 'label', confidence_score, sentiment_score]",
                            f"Observed {entity=} in {entities=} in {text=}",
                        ]
                    )
                )
                continue
            entity_text = entity[0]
            entity_label = entity[1]
            start_idx = text.find(entity_text)
            end_idx = start_idx + len(entity_text)
            if start_idx != -1:
                labels.append([start_idx, end_idx, entity_label])
        spacy_item = {
            "id": str(deterministic_hash(text)),
            "text": text,
            "label": labels,
            "Comments": [],
        }
        spacy_data.append(spacy_item)
    return spacy_data
