import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_prediction_records(input_json_file: str) -> List[Dict[str, Any]]:
    """Loads ReXKG prediction records from JSON or JSONL."""
    p = Path(input_json_file).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input prediction file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported prediction format in {p}")


def _resolve_candidates(doc: Dict[str, Any], key_pred: str, key_gold: str) -> Sequence[Any]:
    pred = doc.get(key_pred, [[]])
    if pred and isinstance(pred, list) and len(pred) > 0 and pred[0]:
        return pred[0]

    gold = doc.get(key_gold, [[]])
    if gold and isinstance(gold, list) and len(gold) > 0:
        return gold[0]
    return []


def rexkg_reverse_structure(input_json_file: str, save_json_file: str) -> List[Dict[str, Any]]:
    """Converts ReXKG relation prediction records into flattened JSON output.

    This mirrors src/ner/result/run_relation/reverse_structure_data.py behavior,
    but is exposed as a reusable PyHealth metrics utility.
    """
    data = _load_prediction_records(input_json_file)

    processed_data: List[Dict[str, Any]] = []
    for doc in data:
        sentence_tokens = (doc.get("sentences") or [[]])[0]
        sentence_text = " ".join(sentence_tokens)

        ner_candidates = _resolve_candidates(doc, "predicted_ner", "ner")
        rel_candidates = _resolve_candidates(doc, "predicted_relations", "relations")

        predicted_entities: Dict[str, str] = {}
        for entity_info in ner_candidates:
            if not entity_info or len(entity_info) < 3:
                continue
            start, end, entity_type = int(entity_info[0]), int(entity_info[1]), str(entity_info[2])
            entity_text = " ".join(sentence_tokens[start : end + 1])
            predicted_entities[entity_text] = entity_type

        predicted_relations: List[Dict[str, str]] = []
        for relation_info in rel_candidates:
            if not relation_info or len(relation_info) < 5:
                continue
            start1, end1, start2, end2, relation_type = relation_info
            entity1_text = " ".join(sentence_tokens[int(start1) : int(end1) + 1])
            entity2_text = " ".join(sentence_tokens[int(start2) : int(end2) + 1])
            predicted_relations.append(
                {
                    "source_entity": entity1_text,
                    "target_entity": entity2_text,
                    "type": str(relation_type),
                }
            )

        processed_doc = {
            "doc_key": doc.get("doc_key"),
            "sentences": sentence_text,
            "entities": predicted_entities,
            "relations": predicted_relations,
        }
        processed_data.append(processed_doc)

    save_path = Path(save_json_file).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as output_file:
        json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

    return processed_data
