from pathlib import Path

import yaml

def load_config(config_path):
    """
    Load configuration from a YAML file.
    the config-data contains the following keys:
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return config