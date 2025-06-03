from .Llama import Llama
from .AzureGPT import AzureGPT
from .Mistral import Mistral
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path, device=None):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'llama':
        model = Llama(config, device)
    elif provider == "azure":
        model = AzureGPT(config, device)
    elif provider == "mistral":
        model = Mistral(config, device)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model