import gc
import random

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, set_seed
import torch
from pydantic import BaseModel, ConfigDict
from typing import Optional, Union, Any, List
from pathlib import Path

from basejenerator.generator_output import GeneratorOutput
from basejenerator.artifacts.text_artifact import TextArtifact
from textjenerator.registry import register
from textjenerator.core.text_generator import BaseTextGenerator


@register("transformers")
class Transformers(BaseTextGenerator):
    """
    Concrete implementation of TextGenerator using Transformers.

    It implements the core abstract methods:
    1. create_pipeline: Initializes the pipeline.
    2. run_pipeline: Executes text generation via the create_chat_completion API.
    """

    def __init__(self, config):
        """
        Initializes the text generator.

        Args:
            config (dict): Configuration dictionary. Must include standard TextGenerator
                           keys plus model-specific keys.
        """
        super().__init__(config)
        self.model = None
        self.config["pretrained_model_name_or_path"] = self.config["model_path"]
        if "seed" in config:
            self.seed = config["seed"]
        else:
            self.seed = self.create_random_seed()
        set_seed(self.seed)


    @staticmethod
    def create_random_seed(size: int = 32) -> int:
        """
        Generates a random integer to serve as a seed.

        Args:
            size (int, optional): The bit-size for the random range. Defaults to 32.

        Returns:
            int: A random integer in the range [0, 2**size - 1].
        """
        seed = random.randint(0, (2**size) - 1)
        return seed


    def load(self):
        """
        Loads the pipeline using AutoModelForCausalLM.from_pretrained and applies config.

        Requires the following keys in self.config:
        * torch_dtype
        * trust_remote_code
        * device_map

        Raises:
            KeyError: If specific config keys are missing.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"],
            local_files_only=True
        )

        if "bnb_config" in self.config.keys():
            bnb_config_params = self.config["bnb_config"]
            if "bnb_4bit_compute_dtype" in bnb_config_params:
                bnb_config_params["bnb_4bit_compute_dtype"] = self.DTYPES_MAP[
                    bnb_config_params["bnb_4bit_compute_dtype"]
                ]

            bnb_config = BitsAndBytesConfig(
                **bnb_config_params
            )
        else:
            bnb_config = None

        self.config["quantization_config"] = bnb_config

        ModelLoadParams = self.get_model_load_params()
        model_load_params = ModelLoadParams(
            **self.config
        )
        model_load_params = model_load_params.model_dump(exclude_none=True)
        model_load_params["quantization_config"] = bnb_config


        self.model = AutoModelForCausalLM.from_pretrained(
            **model_load_params
        )


    def prepare(self):
        """
        """
        set_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    def generate_impl(self):
        """
        Runs inference.
        """
        import time
        start_time = time.time()

        inputs = self.tokenizer.apply_chat_template(
            self.config["messages"],
            return_tensors = "pt",
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            padding_side = "left",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config["max_new_tokens"],
            pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
            do_sample=self.config["do_sample"],
            return_dict_in_generate=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"],
        )

        input_token_count = inputs.input_ids.shape[1]
        output_token_count = output.sequences.shape[1]
        new_tokens = output.sequences[0][input_token_count:]
        new_tokens_generated = output_token_count - input_token_count

        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if not output_text:
            output_text = "[No response generated.]"

        end_time = time.time()
        time_taken = end_time - start_time
        item_extras = {
            "seed": self.seed,
            "input_token_count": input_token_count,
            "output_token_count": new_tokens_generated,
        }

        artifacts = self._quick_wrap([output_text], [item_extras], TextArtifact)

        return GeneratorOutput(artifacts)


    def teardown(self):
        """
        Deletes the pipeline, empties the torch cache, and forces Python's garbage collector to run. Clears the slate to create another pipeline.
        """

        if self.model is None:
            print("No pipeline found. You cannot teardown that which was not created.")
            return

        del self.model
        self.model = None

        gc.collect()


    def get_runtime_params(self) -> set[str]:
        """
        Returns parameters in the model that, if changed, DO NOT require a teardown and 
        reload of the model.

        Returns:
            Set[str]: A set containing the names of the parameters.     
        """
        return (
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "messages"
        )
    
    
    def get_params_schema(self):
        class ParamsSchema(BaseModel):
            backend: str = "transformers",
            model_path: str = "meta-llama/Llama-3.2-3B-Instruct",
            trust_remote_code: bool =  False,

            dtype: str = "float16"
            device_map: str = "cuda"
            
            max_new_tokens: int = 256
            do_sample: bool = True
            temperature: float = 0.1
            top_p: float = 0.8
            top_k: int = 40

            messages: list[str] = []

        return ParamsSchema

    
    def get_model_load_params(self):
        class ModelLoadParams(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            pretrained_model_name_or_path: Union[str, Path] = ""
            torch_dtype: Union[str, torch.dtype] = "auto"
            trust_remote_code: bool = False
            local_files_only: bool = True
            device_map: str = "cuda"
            attn_implementation: str = "sdpa"

        return ModelLoadParams