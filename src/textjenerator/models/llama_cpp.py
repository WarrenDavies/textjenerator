from llama_cpp import Llama
import time
import datetime 

from textjenerator.models.registry import register_model
from textjenerator.core.text_generator import TextGenerator


@register_model("llama-cpp")
class LlamaCPP(TextGenerator):
    """
    Concrete implementation of TextGenerator for Llama CPP.

    """

    def __init__(self, config):
        """
        Initializes the <model name> generator.

        Args:
            config (dict): Configuration dictionary. Must include standard BaseClass
                           keys plus model-specific keys.
        """
        super().__init__(config)
        self.llm = None


    def create_pipeline(self):
        """
        Loads the pipeline and applies configurations.

        Steps taken:
        1. Loads the pipeline using <class>
        2. 

        Raises:
            KeyError: If specific config keys (like 'model_path') are missing.
        """
        self.llm = Llama(
            model_path=self.config["model_path"],
            n_ctx=self.config["max_context_size"],
            n_threads=self.config["number_of_threads"],
            verbose=self.config["verbose_warnings"],
        )


    def run_pipeline_impl(self):
        """
        Executes the inference.
        """
        output = self.llm.create_chat_completion(
            messages=self.config["messages"],
            max_tokens=self.config["max_tokens_per_response"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"],
        )
        choice = output["choices"][0]["message"]
        output_text = (choice.get("content") or "").strip()
        if not output_text:
            output_text = "[No response generated.]"
        self.response = output_text


    def complete_generation_record_impl(self):
        """
        Implementation hook for recording extra stats.
        """
        pass

