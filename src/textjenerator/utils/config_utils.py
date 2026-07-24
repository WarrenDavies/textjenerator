import yaml


def read_yaml_to_dict(file_path):
    """
    Reads a YAML file and returns its content as a dictionary.
    
    Args:
        file_path (str): The path to the .yaml or .yml file.
        
    Returns:
        dict: The parsed YAML content.
    """
    try:
        with open(file_path, 'r') as file:
            # safe_load is used to avoid executing arbitrary code
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None