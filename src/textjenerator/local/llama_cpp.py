from llama_cpp import Llama
from basejenerator.generator_output import GeneratorOutput
from basejenerator.artifacts.text_artifact import TextArtifact
from textjenerator.registry import register_model
from textjenerator.core.text_generator import BaseTextGenerator


@register_model("llama-cpp")
class LlamaCPP(BaseTextGenerator):
    """
    Concrete implementation of TextGenerator using the llama-cpp-python library
    for running GGUF-formatted LLMs (e.g., Llama 3, Mistral) efficiently on CPU.

    It implements the core abstract methods:
    1. create_pipeline: Initializes the llama_cpp.Llama object.
    2. run_pipeline: Executes text generation via the create_chat_completion API.
    """

    def __init__(self, config):
        """
        Initializes the text generator.

        Args:
            config (dict): Configuration dictionary. Must include standard TextGenerator
                           keys plus model-specific keys.
            llm (llama_cpp.Llama | None): The initialized Llama CPP model object.
        """
        super().__init__(config)
        self.llm = None


    def load(self):
        """
        Loads the pipeline using llama_cpp.Llama and applies configurations.

        Note that llama-cpp-python handles its own hardware acceleration, so the device
        config (cpu/CUDA) in the parent class is not used here.

        Requires the following keys in self.config:
        * model_path (str): Path to the GGUF model file.
        * max_context_size (int): The context window size (n_ctx).
        * number_of_threads (int): The number of threads to use (n_threads).
        * verbose_warnings (bool): Enable/disable verbose warnings.
        * n_gpu_layers (int): Number of layers to offload to the GPU (optional, defaults to 0).

        Raises:
            KeyError: If specific config keys (like model_path) are missing.
        """
        self.llm = Llama(
            model_path=self.config["model_path"],
            n_ctx=self.config["max_context_size"],
            n_threads=self.config["number_of_threads"],
            verbose=self.config["verbose_warnings"],
            n_gpu_layers=self.config["n_gpu_layers"]
        )


    def prepare(self):
        """
        No-op for Llama-cpp
        """
        pass


    def generate_impl(self):
        """
        Executes the inference using the chat completion API.

        The final generated text is stored in self.response and returned.
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

        artifacts = self._quick_wrap([output_text], [{}], TextArtifact)
    
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