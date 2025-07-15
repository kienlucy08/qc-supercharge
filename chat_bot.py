'''
File: chat_boy.py
Author: Lucy Kien
Date: 07/15/2025

Python module to run a conversational chatbot for querying QC field rules from LanceDB.
'''

# Imports
import os
import pandas as pd
from openai import OpenAI
import lancedb
from sentence_transformers import SentenceTransformer
from query_database import get_field_value_from_json, find_null_like_fields_from_table, find_null_like_fields, connect_to_collection, query_collection

# Initialize clients
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chatbot methods
# ---------------

def get_bot_instructions(table, limit=15):
    instruction_rows = (
        table.search(embedding_model.encode("bot_instruction").tolist())
        .where("field_name = 'bot_instruction'")
        .limit(limit)
        .to_pandas()
    )
    return [row["bot_response"] for _, row in instruction_rows.iterrows()]

def summarize_with_gpt(user_query, results_df, instructions):
    instruction_block = "\n".join([f"- {line}" for line in instructions])
    context_rows = "\n".join([
        f"- Field: {r['field_name']}\n  Response: {r['bot_response']}\n  Required: {r['required']}\n  Format: {r['expected_format']}\n  Type: {r.get('field_key_type', 'N/A')}"
        for r in results_df.to_dict(orient="records")
    ])
    prompt = f"""
    You are a QC assistant trained to validate tower inspection fields. Follow these core bot instructions:

    {instruction_block}

    A user asked:
    "{user_query}"

    Based on the top-matching fields below, give a clear, grounded, professional answer. Flag anything missing, invalid, or unknown.

    {context_rows}

    Answer:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def introduction(table):
    instructions = get_bot_instructions(table)
    prompt = "Introduce yourself as a QC assistant."
    context = "\n".join([f"- {line}" for line in instructions])
    full_prompt = f"{context}\n{prompt}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.4,
        max_tokens=300
    )
    print(f"QC Assistant Bot: {response.choices[0].message.content.strip()}")

def conclude():
    print("\nQC Assistant Bot: Thanks for chatting. Good luck with your inspections! 🛠️")

# Run chatbot interactively
# -------------------------

def run_qc_chatbot():
    table = connect_to_collection()
    introduction(table)
    context = []

    while True:
        print()
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            conclude()
            break

        if "null values" in user_input.lower() or "fields that are null" in user_input.lower():
            results_df = find_null_like_fields_from_table(table)
            if not results_df:
                print("QC Assistant Bot: I couldn't find any fields with null values.")
            else:
                print("\nQC Assistant Bot: Based on the schema, these fields contain null values:")
                for path in results_df:
                    print(f"{path} - {row['field_name']}")
            continue

        if "value" in user_input.lower() or "what is" in user_input.lower():
            # Try to extract field name from input
            for row in results_df.itertuples():
                if row.field_name:
                    field_value = get_field_value_from_json("your_data.json", row.field_name)
                    print(f"\nQC Assistant Bot: The value of '{row.field_name}' is: {field_value}")
                    context.append(f"The value of '{row.field_name}' is: {field_value}")
                    break
            continue

        results_df = query_collection(table, user_input)
        if results_df.empty:
            print("QC Assistant Bot: Sorry, I couldn't find anything related to that question. Try asking about a specific field.")
            continue

        answer = summarize_with_gpt(user_input, results_df, get_bot_instructions(table))
        print(f"\nQC Assistant Bot: {answer}")
        context.append(user_input)
        context.append(answer)

# Main function
if __name__ == "__main__":
    run_qc_chatbot()
