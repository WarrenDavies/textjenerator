from typing import Union
from datetime import datetime
import yaml
import json
import textwrap
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

def create_app(generator) -> FastAPI:

    app = FastAPI()

    app.state.text_generator = generator

    @app.post("/generate")
    # @app.post("/v1/chat/completions")
    # @app.post("/v1/completions")
    # @app.post("models")
    def generate(request: Request, message: dict):
        text_generator = request.app.state.text_generator
        text_generator.config["messages"] = message["messages"]
        output = text_generator.generate()
        response = output.batch[0].data
        text_generator.config["messages"] = []
        return response


    def stream_llm_tokens(response: str):
        chunk_id = "chatcmpl-local-123"
        created_time = int(time.time())
        
        # Optional: Initial chunk announcing assistant role
        first_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "my-local-model",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        tokens = response.split()
        for token in tokens:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "my-local-model",
                "choices": [{"index": 0, "delta": {"content": token + " "}, "finish_reason": None}]
            }
            # MUST format as data: {json}\n\n
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk indicating finish
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "my-local-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
        # Signal end of stream
        yield "data: [DONE]\n\n"


    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat_completions(request: Request, message: dict):
        print(message)
        text_generator = request.app.state.text_generator
        text_generator.config["messages"] = message["messages"]
        output = text_generator.generate()
        response = output.batch[0].data
        text_generator.config["messages"] = []
        return StreamingResponse(
            stream_llm_tokens(response), 
            media_type="text/event-stream"
        )


    return app