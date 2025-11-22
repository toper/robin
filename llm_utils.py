from config import LMSTUDIO_BASE_URL
from typing import Callable, Optional, List
import requests
from urllib.parse import urljoin
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler


class BufferedStreamingHandler(BaseCallbackHandler):
    def __init__(self, buffer_limit: int = 60, ui_callback: Optional[Callable[[str], None]] = None):
        self.buffer = ""
        self.buffer_limit = buffer_limit
        self.ui_callback = ui_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer += token
        if "\n" in token or len(self.buffer) >= self.buffer_limit:
            print(self.buffer, end="", flush=True)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""

    def on_llm_end(self, response, **kwargs) -> None:
        if self.buffer:
            print(self.buffer, end="", flush=True)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""


# --- Configuration Data ---
# Instantiate common dependencies once
_common_callbacks = [BufferedStreamingHandler(buffer_limit=60)]

# Define common parameters for most LLMs
_common_llm_params = {
    "temperature": 0,
    "streaming": True,
    "callbacks": _common_callbacks,
}

# Map input model choices (lowercased) to their configuration
# Each config includes the class and any model-specific constructor parameters
_llm_config_map = {
    'gpt-4.1': { 
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-4.1'} 
    },
    'gpt-5.1': { 
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.1'} 
    },
    'gpt-5-mini': { 
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-mini'} 
    },
    'gpt-5-nano': { 
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-nano'} 
    },
    'claude-sonnet-4-5': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-5'}
    },
    'claude-sonnet-4-0': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-0'}
    },
    'gemini-2.5-flash': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash'}
    },
    'gemini-2.5-flash-lite': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash-lite'}
    },
    'gemini-2.5-pro': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-pro'}
    },
    # LM Studio models use ChatOpenAI with custom base_url
    'huihui-qwen3-vl-8b-instruct-abliterated': {
        'class': ChatOpenAI,
        'constructor_params': {'model': 'huihui-qwen3-vl-8b-instruct-abliterated', 'base_url': LMSTUDIO_BASE_URL, 'api_key': 'lm-studio'}
    },
    # Add more local models here easily:
    # 'local-model-name': {
    #     'class': ChatOpenAI,
    #     'constructor_params': {'model': 'model-identifier', 'base_url': LMSTUDIO_BASE_URL, 'api_key': 'lm-studio'}
    # }
}


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def _get_lmstudio_base_url() -> Optional[str]:
    if not LMSTUDIO_BASE_URL:
        return None
    return LMSTUDIO_BASE_URL.rstrip("/")


def fetch_lmstudio_models() -> List[str]:
    """
    Retrieve the list of locally available LM Studio models by querying the LM Studio HTTP API.
    Returns an empty list if the API isn't reachable or the base URL is not defined.
    """
    base_url = _get_lmstudio_base_url()
    if not base_url:
        return []

    try:
        # LM Studio uses OpenAI-compatible /v1/models endpoint
        resp = requests.get(f"{base_url}/models", timeout=3)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        available = []
        for m in data:
            model_id = m.get("id")
            if model_id:
                available.append(model_id)
        return available
    except (requests.RequestException, ValueError):
        return []


def get_model_choices() -> List[str]:
    """
    Combine the statically configured cloud models with the locally available LM Studio models.
    """
    base_models = list(_llm_config_map.keys())
    dynamic_models = fetch_lmstudio_models()

    normalized = {_normalize_model_name(m): m for m in base_models}
    for dm in dynamic_models:
        key = _normalize_model_name(dm)
        if key not in normalized:
            normalized[key] = dm

    # Preserve the order: original base models first, then the dynamic ones in alphabetical order
    ordered_dynamic = sorted(
        [name for key, name in normalized.items() if name not in base_models],
        key=_normalize_model_name,
    )
    return base_models + ordered_dynamic


def resolve_model_config(model_choice: str):
    """
    Resolve a model choice (case-insensitive) to the corresponding configuration.
    Supports both the predefined remote models and any locally installed LM Studio models.
    """
    model_choice_lower = _normalize_model_name(model_choice)
    config = _llm_config_map.get(model_choice_lower)
    if config:
        return config

    # Check if it's a LM Studio model
    for lmstudio_model in fetch_lmstudio_models():
        if _normalize_model_name(lmstudio_model) == model_choice_lower:
            return {
                "class": ChatOpenAI,
                "constructor_params": {
                    "model": lmstudio_model,
                    "base_url": LMSTUDIO_BASE_URL,
                    "api_key": "lm-studio"  # LM Studio doesn't require a real API key
                },
            }

    return None
