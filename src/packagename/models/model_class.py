from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
import time
import datetime 
from packagename.models.registry import register_model

from packagename.core.base_class import BaseClass


@register_model("model_name")
class ModelClass(BaseClass):
    """
    Concrete implementation of BaseClass for <model name>.

    """

    def __init__(self, config):
        """
        Initializes the <model name> generator.

        Args:
            config (dict): Configuration dictionary. Must include standard BaseClass
                           keys plus model-specific keys.
        """
        super().__init__(config)
        self.pipe = None


    def create_pipeline(self):
        """
        Loads the pipeline and applies configurations.

        Steps taken:
        1. Loads the pipeline using <class>
        2. 

        Raises:
            KeyError: If specific config keys (like 'model_path') are missing.
        """
        self.pipe = ModelClass.model_method(
            self.config["model_path"],
            torch_dtype=self.dtype,
        ).to(self.device)


    def run_pipeline_impl(self):
        """
        Executes the inference.
        """
        self.output = self.pipe(
            ...
        )


    def complete_generation_record_impl(self):
        """
        Implementation hook for recording extra stats.
        """
        pass

