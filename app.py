from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
import uuid

from preload_database_lance import create_or_reset_collection, preload_fields_from_json, load_bot_instructions
from chat_bot import summarize_with_gpt, query_nullable_fields, query_required_fields, query_all_field_info, get_bot_instructions

app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv("APP_KEY")
UPLOAD_FOLDER = 'uploaded'
SESSION_KEY = os.getenv("SESSION_KEY", "uploaded_file")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global table
table = None

@app.before_request
def clear_session_on_first_visit():
    if "visited" not in session:
        session.clear()
        session["visited"] = True

@app.route("/", methods=["GET", "POST"])
def index():
    global table

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".json"):
            # Save file
            filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.json")
            file.save(filepath)

            # Save the uploaded filename to session
            session["filename"] = file.filename
            session[SESSION_KEY] = filepath
            session["uploaded"] = True

            # Preload database
            table = create_or_reset_collection()
            with open(filepath, "r") as f:
                payload = json.load(f)
                preload_fields_from_json(table, payload)
                load_bot_instructions(table)

            return redirect(url_for("chat"))

    return render_template("index.html", uploaded=session.get("uploaded"), filename=session.get("filename"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global table

    questions = {
        "1": "Show all fields with null or placeholder values",
        "2": "What required fields are missing from this payload?",
        "3": "List all expected fields with their types and categories",
        "4": "Summarize potential data quality issues",
    }

    answer = ""
    selected = ""

    # Rebuild table if not available (e.g., on fresh request)
    if not table and SESSION_KEY in session:
        filepath = session[SESSION_KEY]
        table = create_or_reset_collection()
        with open(filepath, "r") as f:
            payload = json.load(f)
            preload_fields_from_json(table, payload)
            load_bot_instructions(table)

    if request.method == "POST":
        selected = request.form.get("question")

        if not table:
            answer = "No JSON has been uploaded yet."
        else:
            instructions = get_bot_instructions(table)
            if selected == "1":
                results_df = query_nullable_fields(table)
                answer = summarize_with_gpt(questions[selected], results_df, instructions)
            elif selected == "2":
                results_df = query_required_fields(table)
                answer = summarize_with_gpt(questions[selected], results_df, instructions)
            elif selected == "3":
                results_df = query_all_field_info(table)
                answer = summarize_with_gpt(questions[selected], results_df, instructions)
            elif selected == "4":
                nulls = query_nullable_fields(table)
                missing = query_required_fields(table)
                combined = nulls._append(missing, ignore_index=True)
                answer = summarize_with_gpt(questions[selected], combined, instructions)

    return render_template("chat.html", uploaded=True, questions=questions, selected=selected, answer=answer, filename=session.get("filename"))

