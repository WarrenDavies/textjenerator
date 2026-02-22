import copy
from abc import ABC, abstractmethod

import torch
from basejenerator.base_generator import BaseGenerator


class BaseTextGenerator(BaseGenerator):
    """
    Abstract base class for text generation. This class handles the generic configuration and execution flow and manages device (CPU/CUDA) /data type (e.g., bfloat16) setup.
    
    Subclasses must implement create_pipeline() and run_pipeline()

    Attributes:
        config (dict): Configuration dictionary containing model parameters, paths, and settings. 
        DTYPES_MAP (dict): A mapping from string names (e.g., "bfloat16") to torch.dtype objects.
        Add: pipe (Any): The initialized model pipeline (to be set by subclasses).
        Add: response (str/Any): The generated text or model output (to be set by subclasses).
    """

    def __init__(self, config):
        """
        Initializes the object with a config.

        Args:
            config (dict): A dictionary containing configuration parameters. Default config comes from textjenerator.config
        """
        self.config = config
        self.pipe = None
        self.response = None
        self.dtype = None
        self.device = None
        self.DTYPES_MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.detect_device_and_dtype()
        self.batch_size = 1


    def get_model_name(self, config):
        """
        """
        return ""


    def process_config(self, config):
        if "model_name" not in config:
            config["model_name"] = self.get_model_name(config)


    def detect_device_and_dtype(self):
        """
        If 'device' or 'dtype' in config are set to "detect", this method attempts
        to choose the optimal settings based on hardware availability (e.g., CUDA).
        This method modifies self.device and self.dtype based on hardware availability and configuration settings.
        """
        if self.config["device"] == "detect":
            self.set_device()
        else:
            self.device = self.config["device"]

        self.set_dtype()
        

    def set_device(self):
        """
        Sets the computation device based on CUDA availability.

        Sets `self.device` to 'cuda' if available, otherwise defaults to 'cpu'.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


    def set_dtype(self):
        """
        Sets the torch data type based on the device and configuration.

        If config['dtype'] is "detect":
            - Sets to torch.bfloat16 if device is 'cuda'.
            - Sets to torch.float32 otherwise.
        Otherwise, maps the string config to the actual torch.dtype object in self.DTYPES_MAP.
        """
        if self.config["dtype"] == "detect":
            if self.device == "cuda":
                self.dtype = torch.bfloat16
                self.config["dtype"] = "bfloat16"
            else:
                self.dtype = torch.float32
                self.config["dtype"] = "float32"
            return
        
        self.dtype = self.DTYPES_MAP[self.config["dtype"]]


    def merge_config(self, config):
        merged_config = copy.deepcopy(self.config)
        merged_config.update(config)

        return merged_config


    @abstractmethod
    def load(self):
        """
        Abstract method to initialize the model pipeline.
        
        Subclasses must implement this to load the specific model and tokenizer/pipeline object, assigning it to self.pipe (e.g., a Hugging Face Pipeline object).
        """
        pass


    @abstractmethod
    def prepare(self):
        """
        Reset lifecycle without tearing down the model - e.g., clear cache, etc.
        """
        pass


    def generate_impl(self):
        """
        The public API that runs inference.

        Returns:
            GeneratorOutput        
        """
        pass


    @abstractmethod
    def teardown(self):
        """
        Deletes the pipeline, empties the torch cache, and forces Python's garbage collector to run. Clears the slate to create
        another pipeline.
        """
        

    