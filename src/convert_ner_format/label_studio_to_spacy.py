from src.utils.id.idfy import deterministic_hash


def convert_label_studio_to_spacy_format(source_data):
    """
    Label Studio format:
        {
          "id": 2,
          "annotations": [
            {
              "id": 1,
              "completed_by": 1,
              "result": [
                {
                  "value": {
                    "start": 15,
                    "end": 51,
                    "text": "looks out over beautiful green lawns",
                    "labels": [
                      "VIEW"
                    ],
                  },
                  "id": "929ab575-1749-496f-8379-fc3f5d2291b3",
                  "from_name": "label",
                  "to_name": "text",
                  "type": "labels",
                  "origin": "prediction",
                },
                {
                  "value": {
                    "start": 59,
                    "end": 97,
                    "text": "Hudson River and the Statue of Liberty",
                    "labels": [
                      "VIEW"
                    ],
                  },
                  "id": "ab805581-7b2c-4d50-b9d3-461afc8bf341",
                  "from_name": "label",
                  "to_name": "text",
                  "type": "labels",
                  "origin": "prediction",
                },
              ],
              "was_cancelled": False,
              "ground_truth": False,
              "created_at": "2024-07-08T10:19:53.224346Z",
              "updated_at": "2024-07-08T10:19:53.224373Z",
              "draft_created_at": None,
              "lead_time": 7.335,
              "prediction": {},
              "result_count": 0,
              "unique_id": "3bc9c7a8-5fe8-40e9-b785-9320564f903b",
              "import_id": None,
              "last_action": None,
              "task": 2,
              "project": 1,
              "updated_by": 1,
              "parent_prediction": 1997,
              "parent_annotation": None,
              "last_created_by": None,
            }
          ],
          "file_upload": "01c1a5c8-llm_extract_output_20240706095040_100_sample_label_studio.json",
          "drafts": [],
          "predictions": [
            1997
          ],
          "data": {
            "text": "The restaurant looks out over beautiful green lawns to the Hudson River and the Statue of Liberty."
          },
          "meta": {},
          "created_at": "2024-07-08T10:18:46.611778Z",
          "updated_at": "2024-07-08T10:19:53.299292Z",
          "inner_id": 1,
          "total_annotations": 1,
          "cancelled_annotations": 0,
          "total_predictions": 1,
          "comment_count": 0,
          "unresolved_comment_count": 0,
          "last_comment_updated_at": None,
          "project": 1,
          "updated_by": 1,
          "comment_authors": [],
        }

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
    target_data = []

    for entry in source_data:
        text = entry["data"]["text"]
        labels = []

        for prediction in entry["annotations"]:
            for result in prediction["result"]:
                entity_text = result["value"]["text"]
                entity_label = result["value"]["labels"][0]
                start_idx = text.find(entity_text)
                end_idx = start_idx + len(entity_text)
                if start_idx != -1:
                    labels.append([start_idx, end_idx, entity_label])

        text_data = {
            "id": str(deterministic_hash(text)),
            "text": text,
            "labels": labels,
            "Comments": [],
        }

        target_data.append(text_data)

    return target_data
