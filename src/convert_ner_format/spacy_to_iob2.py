from loguru import logger
import spacy

from src.utils.time.timer import log_time

# Load a SpaCy model
nlp = spacy.load("en_core_web_sm")


@log_time(printer=logger.info)
def convert_from_spacy_to_iob2(data):
    results = []
    for item in data:
        text = item["text"]
        labels = item["label"]

        # Create a SpaCy Doc object
        doc = nlp(text)

        # Initialize IOB2 tags
        iob2_tags = ["O"] * len(doc)

        # Assign IOB2 tags based on the labels
        for label in labels:
            start_char = label[0]
            end_char = label[1]
            entity_label = label[2]

            for token in doc:
                if token.idx >= start_char and token.idx < end_char:
                    if token.idx == start_char:
                        iob2_tags[token.i] = f"B-{entity_label}"
                    else:
                        iob2_tags[token.i] = f"I-{entity_label}"

        # Collect the tokens and their IOB2 tags
        token_tags = [(token.text, tag) for token, tag in zip(doc, iob2_tags)]
        results.append(token_tags)

    return results


def add_metadata(iob2_data, spacy_data):
    """
    Add metadata to iob2_data to make the format identical to ConLL2003 data from HuggingFace Fine-tuning Token Classification tutorial
    """
    outputs = []
    for i in range(len(iob2_data)):
        tokens, tags = zip(*iob2_data[i])
        output = {
            **{k: v for k, v in spacy_data[i].items() if k != "label"},
            "tokens": tokens,
            "ner_tags": tags,
        }
        outputs.append(output)
    return outputs


def build_ner_tags_label(iob2_data):
    ner_tags_set = set()
    for record in iob2_data:
        ner_tags_set.update(list(record["ner_tags"]))
    ner_tags_set.remove("O")

    def key_sort_ner_tags(tag):
        """Convert from B-FOOD to FOOD-B, I-FOOD to FOOD-I"""
        pre, suf = tag.split("-")
        return f"{suf}-{pre}"

    ner_tags_label = sorted(list(ner_tags_set), key=key_sort_ner_tags)
    ner_tags_label.insert(0, "O")
    return ner_tags_label
