from abc import ABC, abstractmethod
import datetime
import time
import os
import random
import torch

from packagename.core.base_class_record import BaseClassRecord
from packagename.config import config


class BaseClass(ABC):
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
        self.generation_record = BaseClassRecord()


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


    @staticmethod
    def create_random_seed(size: int = 32) -> int:
        """
        Generates a random integer to serve as a seed.

        Args:
            size (int, optional): The bit-size for the random range. Defaults to 32.

        Returns:
            int: A random integer in the range [0, 2**size - 1].
        """
        return random.randint(0, (2**size) - 1)


    @abstractmethod
    def create_pipeline(self):
        """
        Abstract method to initialize the model pipeline.
        
        Subclasses must implement this to load the specific model
        and assign it to `self.pipe`.
        """
        pass


    def run_pipeline(self):
        """
        Executes the pipeline implementation and tracks performance metrics.

        Calculates run time and ... , updating
        the `generation_record`.
        """
        start_time = time.time()
        self.run_pipeline_impl()
        end_time = time.time()
        self.generation_record.total_generation_time = end_time - start_time


    @abstractmethod
    def run_pipeline_impl(self):
        """
        Abstract method containing the core generation logic.

        Subclasses must implement this to call the model pipeline and populate `self.images`.
        """
        pass
    

    def generate_(self):
        """
        #### rename to generate_image, generate_text, generate_speeh... etc.

        Main workflow method to generate and save images.

        Steps:
            1. Creates the pipeline.
            2. Runs the pipeline implementation.
        """
        self.create_pipeline()
        self.run_pipeline()
        self.save()
        if self.config["save_gen_stats"]:
            self.save_gen_stats()


    def save_(self):
        """
        #### rename to save_image, save_text, save_speeh... etc.

        Saves generated XXXX to the configured directory with a timestamped filename.

        """
        self.save_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{self.save_timestamp}.png"
        save_path = os.path.join(self.config["save_folder"], file_name)
        output.save(save_path)


    def complete_generation_record(self):
        """
        Populates the generation record with metadata.

        Args:
        """
        self.generation_record.gen_data_file_path = self.config["gen_data_file_path"]
        self.generation_record.filename = f"{self.save_timestamp}.png"
        self.generation_record.timestamp = self.save_timestamp
        self.generation_record.model = self.config["model"]
        self.generation_record.device = self.device
        self.generation_record.dtype = self.config["dtype"]
        self.complete_generation_record_impl()


    @abstractmethod
    def complete_generation_record_impl(self):
        """
        Abstract hook for subclasses to add model-specific statistics to the record.
        """
        pass


    def save_gen_stats(self):
        """
        Saves metadata to the record file.
        """
        self.complete_generation_record(prompt, i)
        self.generation_record.save_data()

