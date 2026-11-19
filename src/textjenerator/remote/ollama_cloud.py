import gc
import random

from pydantic import BaseModel, ConfigDict
from typing import Optional, Union, Any, List
from pathlib import Path

from basejenerator.generator_output import GeneratorOutput
from basejenerator.artifacts.text_artifact import TextArtifact
from textjenerator.registry import register
from textjenerator.core.text_generator import BaseTextGenerator

from openai import OpenAI

@register("ollama_cloud")
class OllamaCloud(BaseTextGenerator):
    """
    Concrete implementation of TextGenerator using OllamaCloud
    """

    def __init__(self, config):
        """
        Initializes the text generator.
        """
        self.config = config
        if "model_path" in config:
            self.config["model"] = config["model_path"]


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
        Sets up client.
        """
        ModelLoadParams = self.get_model_load_params()
        model_load_params = ModelLoadParams(
            **self.config
        )
        model_load_params = model_load_params.model_dump(exclude_none=True)

        self.model = OpenAI(
            **model_load_params
        )


    def fix_bnb_config(self, bnb_config):
        pass


    def prepare(self):
        """
        """
        pass


    def warmup(self, prompt=None):
        """
        """
        pass


    def generate_impl(self):
        """
        Runs inference.
        """
        import time
        start_time = time.time()

        completion = self.model.chat.completions.create(
            model=self.config["model"],
            messages=self.config["messages"],
        )

        # this is the entire message history
        input_token_count = completion.usage.prompt_tokens

        # this includes reasoning if the model supports it
        output_token_count = completion.usage.completion_tokens

        output_text = completion.choices[0].message.content

        if not output_text:
            output_text = "[No response generated.]"

        end_time = time.time()
        time_taken = end_time - start_time
        item_extras = {
            "seed": None,
            "reasoning": completion.choices[0].message.reasoning,
            "metrics": {
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
            }
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
            "model",
        )
    
    
    def get_params_schema(self):
        class ParamsSchema(BaseModel):
            model: str = "",

        return ParamsSchema

    
    def get_model_load_params(self):
        class ModelLoadParams(BaseModel):
            base_url: str = "https://ollama.com/v1"
            api_key: str = ""

        return ModelLoadParams
