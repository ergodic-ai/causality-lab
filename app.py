import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy
import pandas
from typing import Optional
from causal_discovery_utils.constraint_based import CDLogger
from server.ai import explain_cg
from server.utils import (
    generate_random_data,
    generate_retention_data,
    graph_object_to_edges,
    initialize_fci,
    run_cd,
    edges_to_graph_object,
)
from effekx.DataManager import SCM
import networkx as nx
from pydantic import BaseModel
from server.utils import is_DAG

app = FastAPI()


class Cache:
    def __init__(self, model=None, edges=None, scm=None):
        self.model = model
        self.edges = edges
        self.scm = scm


cache = Cache()

# Allow CORS requests from any location
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Array to store messages
messages = []


def replace_non_compliant_floats(obj):
    """
    Recursively replace out-of-range float values (NaN, Infinity, -Infinity) in a nested structure
    with None (or some other JSON-compliant value).
    """
    if isinstance(obj, float):
        if (
            obj == float("inf") or obj == float("-inf") or obj != obj
        ):  # Checks for Infinity, -Infinity, and NaN
            return None  # Replace with None or another appropriate value
        else:
            return obj
    if isinstance(obj, (int, numpy.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: replace_non_compliant_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_non_compliant_floats(v) for v in obj]
    else:
        return obj


class StreamLogger(CDLogger):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def wrap(self, message_json):
        async def send():
            try:
                await self.websocket.send_json(
                    replace_non_compliant_floats(message_json)
                )
            except WebSocketDisconnect:
                print("Client disconnected")

        asyncio.create_task(send())

    async def log_now(self, msg: str, metadata: Optional[dict] = None) -> None:
        message_json = {"message": msg, "metadata": metadata, "dataType": "log"}
        await self.websocket.send_json(replace_non_compliant_floats(message_json))

    def log(self, msg: str, metadata: Optional[dict] = None) -> None:
        # print(msg, metadata)
        self.wrap({"message": msg, "metadata": metadata, "dataType": "log"})

    def graph(self, graph: dict, scm=None):
        edges = graph_object_to_edges(graph)
        if scm is not None:
            for edge in edges:
                target = edge["target"]
                source = edge["source"]
                edge["data"]["strength"] = scm.get_strength(source, target)

        self.wrap(
            {
                "metadata": edges,
                "dataType": "graph",
                "message": "Sending new causal graph",
            }
        )
        # pass

    def info(self, msg: str, metadata: Optional[dict] = None) -> None:
        metadata = metadata or {}
        metadata = {**metadata, "level": "info"}
        self.log(msg, metadata)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    model = cache.model
    edges = cache.edges
    scm = cache.scm

    try:

        logger = StreamLogger(websocket)

        if model is not None:
            try:
                logger.graph(model.graph._graph, scm)
            except Exception as e:
                print(e)

        while True:
            data = await websocket.receive_json()
            # print(data)
            uuid = data.get("uuid", None)
            if uuid is None:
                await websocket.send_json({"error": "No UUID provided"})
                continue

            df = generate_retention_data()

            message = data.get("message", None)

            if message == "hello":
                model = initialize_fci(df, logger=logger)
                logger.graph(model.graph._graph)

            if message == "run_all":
                if model is None:
                    model = initialize_fci(df, logger=logger)
                # ai_response = explain_cg(run_cd(model))
                # logger.info(ai_response, metadata={})
                edges = graph_object_to_edges(model.graph._graph)
                if is_DAG(edges):
                    message = "fit"
                else:
                    logger.graph(model.graph._graph)

            if message == "iterate":
                if model is None:
                    model = initialize_fci(df, logger=logger)
                running = model.learn_structure_iterative()
                edges = graph_object_to_edges(model.graph._graph)
                if is_DAG(edges):
                    message = "fit"
                else:
                    logger.graph(model.graph._graph)

                if not running:
                    logger.log("Finished.")

                    async def explain_task():
                        ai_response = explain_cg(edges)
                        logger.info(ai_response, metadata={})

                    # asyncio.create_task(explain_task())

            if message == "json":
                edges = data["data"]["edges"]

                graph_obj = edges_to_graph_object(edges)
                if model is None:
                    model = initialize_fci(df, logger=logger)

                model.graph = graph_obj
                if is_DAG(edges):
                    message = "fit"
                else:
                    logger.graph(model.graph._graph)

            if message == "fit":
                edge_list = [(x["source"], x["target"]) for x in edges]
                causal_graph = nx.DiGraph(edge_list)
                scm = SCM(data=df, graph=causal_graph, logger=logger)
                scm.fit_all()
                logger.log("Finished fitting SCM")
                logger.graph(model.graph._graph, scm=scm)

            cache.model = model
            cache.edges = edges
            cache.scm = scm
            print("Cache updated")

            await websocket.send_json({"message": "Message received"})
    except WebSocketDisconnect as e:
        print("Client disconnected")
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
