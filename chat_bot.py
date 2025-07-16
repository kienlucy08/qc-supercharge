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
from query_database import get_field_value_from_json, query_nullable_fields, query_required_missing_fields, connect_to_collection, query_collection

# Initialize clients
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chatbot methods
# ---------------

def query_required_fields(table):
    df = table.to_pandas()
    return df[df["required"] == True]

def query_all_field_info(table):
    df = table.to_pandas()
    return df[["field_name", "expected_format", "field_category", "field_key_type", "required"]]

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
    prompt = "Introduce yourself as a QC assistant. Provide the questions the user can choose from."
    context = "\n".join([f"- {line}" for line in instructions])
    full_prompt = f"{context}\n{prompt}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.4,
        max_tokens=300
    )
    print(f"\nQC Assistant Bot: {response.choices[0].message.content.strip()}")

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

        # Shared instruction set
        bot_instructions = get_bot_instructions(table)

        if user_input == "1":
            results_df = query_nullable_fields(table)
            if results_df.empty:
                print("\nQC Assistant Bot: I couldn't find any fields with null values.")
            else:
                prompt = "List and explain all fields that have null or missing values in the schema."
                answer = summarize_with_gpt(prompt, results_df, bot_instructions)
                print(f"\nQC Assistant Bot: {answer}")
            continue

        elif user_input == "2":
            # Try to extract field name from input
            for row in results_df.itertuples():
                if row.field_name:
                    field_value = get_field_value_from_json("your_data.json", row.field_name)
                    print(f"\nQC Assistant Bot: The value of '{row.field_name}' is: {field_value}")
                    context.append(f"The value of '{row.field_name}' is: {field_value}")
                    break
            continue

        elif user_input == "3":
            missing_df = query_required_missing_fields(table)
            if missing_df.empty:
                print("\nQC Assistant Bot: I couldn't find any fields with null values.")
            else:
                prompt = "Which required fields are missing from the current payload?"
                answer = summarize_with_gpt(prompt, missing_df, bot_instructions)
                print(f"\nQC Assistant Bot: {answer}")
            continue

        elif user_input == "4":
            summary_df = table.to_pandas()
            if summary_df.empty:
                print("\nQC Assistant Bot: I couldn't find any fields with null values.")
            else:
                prompt = "List all fields and summarize their expected data types and categories."
                answer = summarize_with_gpt(prompt, summary_df, bot_instructions)
                print(f"\nQC Assistant Bot: {answer}")
            continue

        elif user_input == "5":
            results_df = table.to_pandas()
            if results_df.empty:
                print("\nQC Assistant Bot: I couldn't find any fields with null values.")
            else:
                prompt = "Summarize any potential data quality issues in this inspection form data."
                answer = summarize_with_gpt(prompt, results_df, bot_instructions)
                print(f"\nQC Assistant Bot: {answer}")
            continue

        else:
            print("Invalid input. Please select a number from 1 to 5 or type 'exit'.")

# Main function
if __name__ == "__main__":
    run_qc_chatbot()
