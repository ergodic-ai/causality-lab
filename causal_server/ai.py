from typing import List
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json

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


def explain_cg(edges: list[dict], scm=None):
    system_message = """You are helping a data scientist understand their data.
    This is a causal graph that was obtained from a causal discovery algorithm.
    Can you create a one-paragraph summary of the graph within the context of the data.
    When showing numbers, round them to 2 significant digits only.
    When speaking about variables in the graph,
    include the original variable name in parenhesis prefixed with node:, for instance:
    the graph shows that sales orders (node:orders) is directly affected by holidays (node:is_holiday).

    """
    if scm is not None:
        for edge in edges:
            target = edge["target"]
            source = edge["source"]
            edge["data"]["strength"] = scm.get_strength(source, target)

        graphstr = " ".join(
            [
                f"{elm['source']} --> {elm['target']} with strength: {elm['data']['strength']}.\n\n"
                for elm in edges
            ]
        )

    else:
        graphstr = " ".join(
            [f"{elm['source']} --> {elm['target']}.\n\n" for elm in edges]
        )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "This is my causal graph: \n" + graphstr},
    ]
    print(messages)
    response = get_openai_response(messages)
    print(response)
    return response


def get_recs(edges: list[dict], scm=None, effects=None):
    system_message = """You are helping a data scientist understand their data and generate recommendations about what to
    do to improve a business. Your objective is to create a list of recommendations. You will see two objects:
     * A causal graph that was obtained from a causal discovery algorithm. Including a quantification of the strength of the edges.
     * A list of effects of the variables in the graph. These contain the source, the target, as well as the paths that contribute
        to the effect, including their contribution to the strength of the effect.
    When showing numbers, round them to 2 significant digits only.
    When speaking about variables in the graph,
    include the original variable name in parenhesis prefixed with node:, for instance:
    the graph shows that sales orders (node:orders) is directly affected by holidays (node:is_holiday).

    """
    if scm is not None:
        for edge in edges:
            target = edge["target"]
            source = edge["source"]
            edge["data"]["strength"] = scm.get_strength(source, target)

        graphstr = " ".join(
            [
                f"{elm['source']} --> {elm['target']} with strength: {elm['data']['strength']}.\n\n"
                for elm in edges
            ]
        )

    else:
        graphstr = " ".join(
            [f"{elm['source']} --> {elm['target']}.\n\n" for elm in edges]
        )

    recsStr = json.dumps(effects)

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": "This is my causal graph: \n"
            + graphstr
            + "\n\nThese are the effects: "
            + recsStr,
        },
    ]
    response = get_openai_response(messages)
    return response
