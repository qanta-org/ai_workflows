"""
Configuration settings for the workflows package.

This module contains configuration settings and constants used across the workflows package,
including model configurations, workflow settings, and other package-wide constants.
"""

AVAILABLE_MODELS = {
    "DeepSeek/V3": {
        "model": "deepseek-chat",
        "logprobs": False,
        "cost_per_million": 0.27,
    },
    "OpenAI/gpt-4.1": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 2.0,
    },
    "OpenAI/gpt-4.1-mini": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 0.4,
    },
    "OpenAI/gpt-4.1-nano": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 0.1,
    },
    "OpenAI/gpt-4o": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 2.50,
    },
    "OpenAI/gpt-4o-mini": {
        "model": "gpt-4o-mini-2024-07-18",
        "logprobs": True,
        "cost_per_million": 0.15,
    },
    "OpenAI/gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo-0125",
        "cost_per_million": 0.15,
    },
    "Anthropic/claude-sonnet-4-6": {
        "model": "claude-sonnet-4-6",
        "cost_per_million": 3.0,
    },
    "Anthropic/claude-haiku-4-5": {
        "model": "claude-haiku-4-5-20251001",
        "cost_per_million": 0.80,
    },
    "Cohere/command-a": {
        "model": "command-a-03-2025",
        "logprobs": True,
        "cost_per_million": 2.50,
    },
    "Cohere/command-r-plus": {
        "model": "command-r-plus-08-2024",
        "logprobs": True,
        "cost_per_million": 2.50,
    },
    "Cohere/command-r": {
        "model": "command-r-08-2024",
        "logprobs": True,
        "cost_per_million": 0.15,
    },
    "Cohere/command-r7b": {
        "model": "command-r7b-12-2024",
        "logprobs": False,
        "cost_per_million": 0.0375,
    },
}

# Function mapping for input/output transformations
TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

FUNCTION_MAP = {
    "upper": str.upper,
    "lower": str.lower,
    "len": len,
    "split": str.split,
}
