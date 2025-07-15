'''
File: preload_database.py
Author: Lucy Kien

Python module to preload the database into a LanceDB collection.
'''

import lancedb
from sentence_transformers import SentenceTransformer
import os
import json
import pandas as pd

# Initialize your embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def infer_field_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "number"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "text"
    elif isinstance(value, list):
        return "list"
    elif isinstance(value, dict):
        return "object"
    return "unknown"

def infer_key_type(field_name):
    fname = field_name.lower()
    if "id" in fname:
        return "identifier"
    elif "time" in fname or "date" in fname:
        return "timestamp"
    elif "email" in fname:
        return "contact"
    return "general"

def flatten_json(obj, parent_key='', sep='.'):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_json(v, new_key, sep=sep))
    else:
        items.append((parent_key, obj))
    return items

def create_or_reset_collection(db_path="./lancedb", collection_name="qc_field_rules"):
    """Create or reset the LanceDB collection."""
    db = lancedb.connect(db_path)

    if collection_name in db.table_names():
        db.drop_table(collection_name)

    table = db.create_table(
        collection_name,
        data=[
            {
                "field_name": "",
                "expected_format": "",
                "validation_type": "",
                "bot_response": "",
                "example_value": "",
                "field_category": "",
                "priority_level": "low",
                "acceptable_values": "",
                "required": False,
                "field_key_type": "",
                "was_null": False,
                "vector": [0.0] * embedding_model.get_sentence_embedding_dimension()
            }
        ],
        mode="overwrite"
    )

    return table

def preload_fields_from_json(table, json_source):
    if isinstance(json_source, str):
        with open(json_source, "r") as f:
            data = json.load(f)
    elif isinstance(json_source, dict):
        data = [json_source]
    elif isinstance(json_source, list):
        data = json_source
    else:
        raise ValueError("Unsupported input type. Must be file path, dict, or list of dicts.")

    if not data:
        print("No data found.")
        return

    flat_fields = flatten_json(data[0])
    rows = []
    seen = set()

    for path, value in flat_fields:
        if path in seen:
            continue
        seen.add(path)

        fmt = infer_field_type(value)
        category = path.split('.')[0]
        key_type = infer_key_type(path)

        rows.append({
            "field_name": path,
            "expected_format": fmt,
            "validation_type": "type_check",
            "bot_response": f"Expected format: {fmt}",
            "example_value": str(value),
            "field_category": category,
            "priority_level": "low",
            "acceptable_values": "",
            "required": False,
            "field_key_type": key_type,
            "was_null": value is None,
            "vector": embedding_model.encode(f"{path} {fmt}").tolist()
        })

    expected_fields = {"attributes.site_visit_datetime", "attributes.customer", "geometry"}
    present_fields = {path for path, _ in flat_fields}
    missing_fields = expected_fields - present_fields

    for m in missing_fields:
        rows.append({
            "field_name": m,
            "expected_format": "unknown",
            "validation_type": "missing_check",
            "bot_response": f"⚠️ Missing expected field: {m}",
            "example_value": "",
            "field_category": m.split('.')[0],
            "priority_level": "high",
            "acceptable_values": "",
            "required": True,
            "field_key_type": "required",
            "was_null": False,
            "vector": embedding_model.encode(f"{m} missing").tolist()
        })

    if rows:
        df = pd.DataFrame(rows)
        table.add(df)
        print(f"✅ Loaded {len(rows)} fields. Categories: {sorted(set(r['field_category'] for r in rows))}")

def load_bot_instructions(table):
    instructions = [
        "Welcome to the QC Assistance Bot! I'm here to help you validate tower inspection forms.",
        "Please answer the user's question with references to required or expected field values.",
        "If a field is missing or seems incorrectly formatted, refer to the correct format and provide an example.",
        "Never respond with hallucinated fields or made-up data. Only use what is stored in the collection.",
        "If the user types 'exit', the session should end.",
        "Categorize fields by their expected data type: text, number, boolean, or timestamp.",
        "If a question is unclear, respond with a clarifying question.",
        "Let the user know if a specific field is not recognized in the schema.",
        "Always use natural, professional, and helpful tone.",
        "Prioritize fields with 'high' priority_level during QA sessions.",
        "Only return information that is defined in the validation schema."
    ]

    rows = [
        {
            "field_name": "bot_instruction",
            "expected_format": "text",
            "validation_type": "guidance",
            "bot_response": msg,
            "example_value": "",
            "field_category": "meta",
            "priority_level": "high",
            "acceptable_values": "",
            "required": False,
            "field_key_type": "instruction",
            "was_null": False,
            "vector": embedding_model.encode(msg).tolist()
        } for msg in instructions
    ]

    df = pd.DataFrame(rows)
    table.add(df)

def main():
    table = create_or_reset_collection()
    preload_fields_from_json(table, json_source="example.json")
    load_bot_instructions(table)

if __name__ == "__main__":
    main()