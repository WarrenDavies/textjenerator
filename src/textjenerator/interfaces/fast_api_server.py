from typing import Union
from datetime import datetime
import yaml
import json
import textwrap

from fastapi import FastAPI, Request


def create_app(generator) -> FastAPI:

    app = FastAPI()

    app.state.text_generator = generator

    @app.post("/generate")
    def generate(request: Request, message: dict):
        text_generator = request.app.state.text_generator
        text_generator.config["messages"] += message["messages"]
        output = text_generator.generate()
        response = output.batch[0].data
        text_generator.config["messages"] = ""
        return response

    return app