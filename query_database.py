"""
File: query_database.py
Author: Lucy Kien

Python module to query the LanceDB QC field validation rules collection.
"""

import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd
from openai import OpenAI
import os
import json
# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def connect_to_collection(db_path="./lancedb", collection_name="qc_field_rules"):
    """
    Connect to the LanceDB collection.

    Parameters:
        - db_path (str): Path to the LanceDB directory.
        - collection_name (str): Name of the LanceDB table.

    Returns:
        - table (lancedb.table.LanceTable): Opened LanceDB table.
    """
    db = lancedb.connect(db_path)
    return db.open_table(collection_name)

def query_collection(table, user_input, top_k=10):
    """
    Perform a vector similarity search against the LanceDB collection.

    Parameters:
        - table: LanceDB table to query.
        - user_input (str): User's question or query text.
        - top_k (int): Number of top results to return.

    Returns:
        - DataFrame with top matching results.
    """
    query_vector = embedding_model.encode(user_input).tolist()
    results = table.search(query_vector).limit(top_k).to_pandas()
    
    filtered = results[results["field_name"] != "bot_instruction"].head(top_k)
    
    return filtered

def get_field_value_from_json(json_path: str, field_path: str):
    """Given a flattened field path, return its value from the JSON file."""
    import json
    from query_database import flatten_json  # reuse your flattening logic

    try:
        with open(json_path, 'r') as f:
            raw = json.load(f)
        flat = dict(flatten_json(raw[0]))  # single feature
        return flat.get(field_path, "Field not found in source JSON.")
    except Exception as e:
        return f"Error accessing JSON: {str(e)}"

def query_nullable_fields(table):
    """Return all fields with either format 'null' or names indicating nullability."""
    df = table.to_pandas()

    # Filter fields where expected_format is 'null' or field_name includes 'nullable'
    null_df = df[
        (df["expected_format"].str.lower() == "null") |
        (df["was_null"] == True)
    ]
    return null_df

def query_required_missing_fields(table):
    """
    Returns a DataFrame of required fields that are missing from the raw_payload.
    These are defined by:
    - required == True
    - validation_type == 'missing_check'
    """
    df = table.to_pandas()

    # Filter required fields that have missing validation
    missing_df = df[
        (df["required"] == True) &
        (df["validation_type"] == "missing_check")
    ]

    return missing_df[["field_name", "bot_response"]]


def find_null_like_fields(obj, path=""):
    null_like_keys = []

    print(f"\n🔍 DEBUG: Traversing {type(obj)} at path: {path}")

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if value in [None, "null", "Null", "NULL"]:
                null_like_keys.append(new_path)
            else:
                null_like_keys.extend(find_null_like_fields(value, new_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            null_like_keys.extend(find_null_like_fields(item, new_path))
    else:
        print(f"⚠️  Skipping unexpected type at {path}: {type(obj)}")

    return null_like_keys


def find_null_like_fields_from_table(table, payload_field="raw_payload"):
    df = table.to_pandas()
    if payload_field not in df.columns:
        print(f"⚠️ Column '{payload_field}' not found in table.")
        return []

    null_like_paths = []
    for i, row in df.iterrows():
        try:
            payload = json.loads(row[payload_field])
            paths = find_null_like_fields(payload)
            null_like_paths.extend(paths)
        except Exception as e:
            print(f"⚠️ Error parsing JSON in row {i}: {e}")
            continue

    return sorted(set(null_like_paths))

def main():
    table = connect_to_collection()

    query = input("Ask a question about QC field rules (type 'show nulls' to list nullable fields): ")

    if query.lower() == "show nulls":
        results = query_nullable_fields(table)
        print("\n🔍 Fields allowing null or marked as nullable:")
        for idx, row in results.iterrows():
            print(f"\nResult {idx + 1}:")
            print(f"- Field: {row['field_name']}")
            print(f"- Format: {row['expected_format']}")
            print(f"- Bot: {row['bot_response']}")
            print(f"- Required: {row['required']}")
    else:
        results = query_collection(table, query)
        print("\n🔍 Top Matching Fields:")
        for idx, row in results.iterrows():
            print(f"\nResult {idx + 1}:")
            print(f"- Field: {row['field_name']}")
            print(f"- Response: {row['bot_response']}")
            print(f"- Priority: {row['priority_level']}")
            print(f"- Required: {row['required']}")

if __name__ == "__main__":
    main()

