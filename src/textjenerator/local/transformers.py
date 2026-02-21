import gc

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from pydantic import BaseModel

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


    def load(self):
        """
        Loads the pipeline using AutoModelForCausalLM.from_pretrained and applies config.

        Requires the following keys in self.config:
        * 

        Raises:
            KeyError: If specific config keys are missing.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_path"],
            # quantization_config=bnb_config,
            torch_dtype=self.config["dtype"],
            trust_remote_code=self.config["trust_remote_code"],
            device_map=self.config["device"]
        )


    def prepare(self):
        """
        """
        pass


    def generate_impl(self):
        """
        Executes the inference using the chat completion API.

        The final generated text is stored in self.response and returned.
        """
        inputs = self.tokenizer.apply_chat_template(
            self.config["messages"],
            **self.config["tokenizer"]
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
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

        item_extras = {
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
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
        )
    
    
    def get_params_schema(self):
        class ParamsSchema(BaseModel):

            n_ctx: int = 0
            n_threads: int = 0
            verbose: int = 0
            n_gpu_layers: int = 0
            max_tokens: int = 0
            temperature: float = 0
            top_p: int = 0
            top_k: int = 0

        return ParamsSchema