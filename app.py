import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy
import pandas
from typing import Optional
from causal_discovery_utils.constraint_based import CDLogger
from server.ai import explain_cg, get_recs
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

    def report(self, msg: str, metadata: Optional[dict] = None) -> None:
        self.wrap({"message": msg, "metadata": metadata, "dataType": "report"})

    def send_wait(self, msg: str, metadata: Optional[dict] = None) -> None:
        self.wrap({"message": msg, "metadata": metadata, "dataType": "wait"})

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


def test_blocking_fn():
    time.sleep(10)
    return "AI response"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    model = cache.model
    edges = cache.edges
    scm = cache.scm

    async def get_ai_response(edges, scm):
        res = await loop.run_in_executor(None, explain_cg, edges, scm)
        return res

    async def get_ai_recommendations(edges, scm, effects):
        res = await loop.run_in_executor(None, get_recs, edges, scm, effects)
        return res

    try:
        logger = StreamLogger(websocket)

        if model is not None:
            try:
                logger.graph(model.graph._graph, scm)
            except Exception as e:
                print(e)

        while True:
            data = await websocket.receive_json()
            logger.send_wait("Received message. Processing...", metadata={"wait": True})
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
                logger.send_wait("Running causal discovery.", metadata={"wait": True})
                if model is None:
                    model = initialize_fci(df, logger=logger)
                _ = run_cd(model)
                edges = graph_object_to_edges(model.graph._graph)
                if is_DAG(edges):
                    message = "fit_explain"
                else:
                    logger.graph(model.graph._graph)

            if message == "iterate":
                logger.send_wait("Stepping...", metadata={"wait": True})
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
                logger.send_wait("Receiving causal graph...", metadata={"wait": True})
                edges = data["data"]["edges"]

                graph_obj = edges_to_graph_object(edges)
                if model is None:
                    model = initialize_fci(df, logger=logger)

                model.graph = graph_obj
                if is_DAG(edges):
                    message = "fit"
                else:
                    logger.graph(model.graph._graph)

            if message == "optimize":
                logger.send_wait("Optimizing actions...", metadata={"wait": True})
                print("optimize")
                if scm is None:
                    logger.log("No SCM found. Please fit the model first.")

                else:
                    logger.report("Optimization report: blablabla")

            if message.startswith("fit"):
                logger.send_wait("Updating causal model...", metadata={"wait": True})
                edges = graph_object_to_edges(model.graph._graph)
                edge_list = [(x["source"], x["target"]) for x in edges]
                causal_graph = nx.DiGraph(edge_list)
                scm = SCM(data=df, graph=causal_graph, logger=logger)
                # print(scm.data_types)
                scm.fit_all()
                logger.log("Finished fitting SCM")
                logger.graph(model.graph._graph, scm=scm)

            if "explain" in message:
                logger.send_wait("Generating graph summary...", metadata={"wait": True})
                edges = graph_object_to_edges(model.graph._graph)
                ai_response = await get_ai_response(edges, scm)
                logger.report(ai_response, metadata={})
                message = "effects"

            if message == "effects":
                logger.send_wait(
                    "Calculating causal effects...", metadata={"wait": True}
                )
                edges = graph_object_to_edges(model.graph._graph)
                edge_list = [(x["source"], x["target"]) for x in edges]
                causal_graph = nx.DiGraph(edge_list)
                end_node = data.get("data", {}).get("target", None)

                print(end_node)
                if end_node is None:
                    end_nodes = [
                        x
                        for x in causal_graph.nodes()
                        if causal_graph.out_degree(x) == 0
                    ]
                    if len(end_nodes):
                        end_node = end_nodes[0]

                if end_node is not None:
                    scm = SCM(data=df, graph=causal_graph, logger=logger)
                    scm.fit_all()

                    nodes = list(nx.ancestors(causal_graph, end_node))
                    results = []
                    for node in nodes:
                        res = scm.get_total_strength(node, end_node)
                        results.append(res)

                    results = sorted(results, key=lambda x: abs(x["total_strength"]))
                    metadata = {
                        "end_node": end_node,
                        "results": results,
                        "action": "effects",
                    }

                    logger.report("Effects report.", metadata=metadata)

                    ai_response = await get_ai_recommendations(edges, scm, results)
                    logger.report(ai_response, metadata={})

            cache.model = model
            cache.edges = edges
            cache.scm = scm
            logger.send_wait("Finished!", metadata={"wait": False})

    except WebSocketDisconnect as e:
        print("Client disconnected")
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
