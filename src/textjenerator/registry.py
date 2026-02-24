"""
Implements a simple decorator-based model registry for dynamically managing
different model classes.

This system allows concrete subclasses to register themselves
using a specific string key, enabling the application to instantiate the 
correct class based on a configuration setting without explicit imports.
"""

REGISTRY = {}

def register(name):
    """
    A decorator factory used to register a subclass.

    The decorated class is stored in the global REGISTRY dictionary
    under the provided `name`.

    Args:
        name (str): The string key used to reference the model class
                    in the configuration.

    Returns:
        Callable: A decorator function that takes a class and registers it.
    """
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator


def get_class(config):
    """
    Retrieves and instantiates the correct class based on the
    configuration dictionary.

    It looks up the class in REGISTRY using the key found in
    `config['model']`.

    Args:
        config (dict): The configuration dictionary, which must contain a
                       'model' key corresponding to a registered backend name.

    Returns:
        BaseClass: An instantiated object of the registered generator class.

    Raises:
        KeyError: If the value of `config['model']` or `config['backend']` are 
                  not found in the REGISTRY.
    """
    if "backend" in config:
        backend = config["backend"]
    elif "model" in config:
        backend = config["model"]
    else:
        raise KeyError

    Class_ = REGISTRY[backend]
    backend_object = Class_(config)

    return backend_object


def get_model_class(config):
    """
    Deprecated - calls the more generically name get_class
    """
    return get_class(config)

