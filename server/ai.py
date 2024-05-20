from typing import List
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT") or "gpt4turbo",
)
deployment_name = os.getenv("DEPLOYMENT_NAME")


def get_openai_response(
    messages: List[dict], response_mode=None, temperature=0, model=deployment_name
) -> str:
    """Get a response from OpenAI GPT4. The response can be in JSON or plain text.
    variables:
        query: str: The user query
        system_message: str: The system message to be passed to the model
        response_mode: str: The response mode. Can be "json" or "text"

    output:
        The response from OpenAI GPT4
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"} if response_mode == "json" else None,
    )

    return response.choices[0].message.content


def get_embedding(query: str):
    response = client.embeddings.create(
        model="ada", input=query, encoding_format="float"
    )
    return response.data[0].embedding


def get_batch_embedding(queries: list[str]):
    response = client.embeddings.create(
        model="ada", input=queries, encoding_format="float"
    )
    return [x.embedding for x in response.data]


def explain_cg(edges: list[dict]):
    system_message = """You are helping a data scientist understand their data.
    This is a causal graph that was obtained from a causal discovery algorithm.
    Can you create a one-paragraph summary of the graph within the context of the data.
    The data relates to retention for a telco company:

    """
    graphstr = " ".join([f"{elm['source']} --> {elm['target']}.\n\n" for elm in edges])
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": graphstr},
    ]
    print(messages)
    response = get_openai_response(messages)
    print(response)
    return response
