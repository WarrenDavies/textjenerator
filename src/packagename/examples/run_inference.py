from packagename.models import registry
from packagename.config import config

base_class = registry.get_model_class(config)
base_class.generate()

