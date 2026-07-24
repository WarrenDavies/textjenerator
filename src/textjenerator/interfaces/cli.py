import argparse
import uvicorn

from textjenerator.interfaces.fast_api_server import app
from textjenerator import registry


default_config = {
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="",
        help="Path to config file"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )

    parser.add_argument(
        "--endpoint",
        default="input"
    )

    args = parser.parse_args()

    print(args)

    if not args.config:
        config = default_config
    else:
        config = args.config

    text_generator = registry.get_model_class(config)
    text_generator.load()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()