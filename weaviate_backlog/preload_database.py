'''
File: preload_database.py
Author: Lucy Kien

Python module to create preload the database into a collection
'''

import weaviate
import weaviate.classes as wvc
import os
import json

def create_client(weaviate_version = "1.24.10") -> weaviate.WeaviateClient:
    """Create the weaviate client

    Parameters:
        - weaviate_version (str): The version of weaviate

    Returns:
        - clientObject (weaviate.WeaviateClient): The weaviate client
    """
    # create the client
    client = weaviate.connect_to_local(
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        }
    )
    return client


def create_collection(client: weaviate.WeaviateClient, 
                      collection_name: str,
                      embedding_model: str = 'text-embedding-3-small',
                      model_dimensions: int = 512):
    """Create the collection using the client, name, and other modeling information

    Parameters:
        - client (weaviate.WeaviateClient): The Weaviate client.
        - collection_name (str): The name of the collection.
        - embedding_model (str): The model used for text embedding.
        - model_dimensions (int): The model dimensions. 
    
    Returns:
        - collection (weaviate.Collection): A weaviate collection.
    """
     # set collection 
    collection = None
    # see if the collection exists
    if client.collections.exists(collection_name):
        collection = client.collections.delete(collection_name)

    # ceate the collection schema
    collection = client.collections.create(
            name=collection_name,
            # set the properties 
            properties=[
                wvc.config.Property(
                    name="field_name",
                    data_type=wvc.config.DataType.TEXT,
                    description="Name of the field"
                ),
                wvc.config.Property(
                    name="expected_format",  
                    data_type=wvc.config.DataType.TEXT,
                    description="Expected format of the field"
                ),
                wvc.config.Property(
                    name="validation_type",
                    data_type=wvc.config.DataType.TEXT,
                    description="What kind of validation is required"
                ),
                wvc.config.Property(
                    name="bot_response",
                    data_type=wvc.config.DataType.TEXT,
                    description="How the bot should respond if this field is missing or wrong"
                ),
                wvc.config.Property(
                    name="example_value",
                    data_type=wvc.config.DataType.TEXT,
                    description="Example of a correct value"
                ),
                wvc.config.Property(
                    name="field_category",
                    data_type=wvc.config.DataType.TEXT,
                    description="Category of field"
                ),
                wvc.config.Property(
                    name="priority_level",
                    data_type=wvc.config.DataType.TEXT,
                    description="Priority level: low, medium, high, urgent"
                ),
                wvc.config.Property(
                    name="acceptable_values",
                    data_type=wvc.config.DataType.TEXT,
                    description="Comma-seperated list of acceptable values"
                ),
                wvc.config.Property(
                    name="required",
                    data_type=wvc.config.DataType.BOOL,
                    description="Whether this field is requried"
                )
            ],
            # configure
            vector_config=[
                    wvc.config.Configure.Vectorizer.text2vec_openai(
                    model=embedding_model,
                    dimensions=model_dimensions
                )
            ],
            generative_config=wvc.config.Configure.Generative.openai()
        )

    return collection  # Return the name of the created collection

# loading the animal data into the database
def load_validation_rules(client: weaviate.WeaviateClient, collection, data_file: str):
    """Load QC field validation rules into the client collection.

    Parameters:
        - client (weaviate.WeaviateClient): The Weaviate client.
        - collection (weaviate.Collection): The Weaviate collection object.
        - data_file (str): Path to the JSON file containing field rules.

    Returns:
        - data_output: Output of the insertion operation.
    """

    with open(data_file, 'r') as file:
        data = json.load(file)

    validation_objects = []
    for item in data:
        validation_objects.append({
            "field_name": item.get("field_name"),
            "expected_format": item.get("expected_format", ""),
            "validation_type": item.get("validation_type", ""),
            "bot_response": item.get("bot_response", ""),
            "example_value": str(item.get("example_value", "")),
            "field_category": item.get("field_category", ""),
            "priority_level": item.get("priority_level", "low"),
            "acceptable_values": ", ".join(item.get("acceptable_values", [])) if item.get("acceptable_values") else "",
            "required": item.get("required", False)
        })

    return collection.data.insert_many(validation_objects)


# loading the bot instructions
def load_bot_instructions(client: weaviate.WeaviateClient, collection):
    """Load bot instructions into the Weaviate collection.

    Parameters:
        - client (weaviate.WeaviateClient): The Weaviate client.
        - collection_name (str): The name of the collection.

    Returns:
        - data_ouput: Collection with added instructions
    """

    instructions_list = [
        {"field_name": "bot_instruction", "expected_format": "text", "validation_type": "guidance", "bot_response": msg, "example_value": "", "field_category": "meta", "priority_level": "high", "acceptable_values": "", "required": False}
        for msg in [
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
    ]
    
    # Insert bot instructions into the collection
    data_output = collection.data.insert_many(instructions_list)
    return data_output



def main():
    client = create_client()
    try:
        collection = create_collection(client, collection_name="qc_field_rules")
        load_validation_rules(client, collection, data_file="example.json")
        load_bot_instructions(client, collection)
    finally:
        client.close()



if __name__ == "__main__":
    main()