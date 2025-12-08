from abc import ABC, abstractmethod
import datetime
import time
import os
import random

from textjenerator.config import config


class TextGenerator(ABC):
    """
    Abstract base class for...

    This class handles... 
    
    Subclasses must implement...

    Attributes:
        config (dict): Configuration dictionary containing model parameters, paths, and settings.
    """

    def __init__(self, config = config):
        """
        Initializes the object with a config.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Expected keys include:
                - 
        """
        self.config = config
        # self.generation_record = BaseClassRecord()
        self.response = None

    def detect_device_and_dtype(self):
        """
        If 'device' or 'dtype' in config are set to "detect", this method attempts
        to choose the optimal settings based on hardware availability (e.g., CUDA).
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


    @abstractmethod
    def create_pipeline(self):
        """
        Abstract method to initialize the model pipeline.
        
        Subclasses must implement this to load the specific model
        and assign it to `self.pipe`.
        """
        pass


    @abstractmethod
    def run_pipeline(self):
        """
        Executes the pipeline implementation and tracks performance metrics.

        the `generation_record`.
        """

        pass
    

    def generate_text(self):
        """
        Main workflow method to generate and return text.

        Steps:
            1. Creates the pipeline.
            2. Runs the pipeline implementation.
        """
        self.create_pipeline()
        self.run_pipeline()
        return(self.response)

