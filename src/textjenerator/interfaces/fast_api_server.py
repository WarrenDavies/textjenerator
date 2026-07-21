from typing import Union
from datetime import datetime
import yaml
import json
import textwrap

from fastapi import FastAPI

from textjenerator import registry


config = {
    # model
    "backend": "transformers",
    "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
    "trust_remote_code": False,
    "local_files_only": True,
    "attn_implementation": "sdpa",
    
    # hardware/system
    "device": "cuda",
    "dtype": "bfloat16",

    # 
    "bnb_config": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "quant_method": "bitsandbytes_4bit"
    },

    # LLM
    "verbose_warnings": False,
    "max_context_size": 65536,
    "max_new_tokens": 2048,
    "do_sample": True,
    "temperature": .8,
    "top_p": 0.9,
    "top_k": 40,
    "messages": [
          {"role": "system", "content": """You are Jenbot, an expert, helpful, and diligent assistant. You provide the user with accurate answers to their queries. You are polite, friendly, and a little sarcastic."""},
    ]
}

app = FastAPI()

text_generator = registry.get_model_class(config)
text_generator.load()

@app.post("/input")
def read_item(message: dict):
    text_generator.config["messages"] = message["messages"]
    output = text_generator.generate()
    response = output.batch[0].data
    text_generator.config["messages"] = ""
    return response
