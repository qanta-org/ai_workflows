# %%

import base64
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional, Tuple

import cohere
import json_repair
import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.exceptions import OutputParserException
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema
from rich import print as rprint

# Initialize global cache
try:
    from src.envs import CACHE_PATH, LLM_CACHE_REPO
except ImportError:
    logger.error(
        "Either module src.envs not found, or CACHE_PATH or LLM_CACHE_REPO not found, trying to look in environment"
    )
    CACHE_PATH = os.environ.get("LLM_CACHE_PATH", ".")
    LLM_CACHE_REPO = None


from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from .configs import AVAILABLE_MODELS
from .llmcache import LLMCache

set_llm_cache(SQLiteCache(database_path=f"{CACHE_PATH}/langchain.db"))

llm_cache = LLMCache(cache_dir=CACHE_PATH, hf_repo=LLM_CACHE_REPO)

MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_DEFAULT_OPENAI_BASE_URL = "https://us.api.openai.com/v1"


def _resolve_openai_base_url() -> str:
    """Base URL for OpenAI SDK / LangChain (must end with /v1).

    Regional API keys return 401 ``incorrect_hostname`` when called via the global
    ``https://api.openai.com/v1``. Hugging Face and other templates often set that host
    explicitly; rewrite only the global hostname so US-pinned projects work.
    """
    from urllib.parse import urlparse

    raw = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "").strip().strip('"').strip("'")
    if not raw:
        return _DEFAULT_OPENAI_BASE_URL
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.warning(f"Invalid OPENAI_BASE_URL {raw!r}; using {_DEFAULT_OPENAI_BASE_URL}")
        return _DEFAULT_OPENAI_BASE_URL
    host = (parsed.hostname or "").lower()
    if host == "api.openai.com":
        logger.info(
            "OPENAI_BASE_URL uses global api.openai.com; using {}. "
            "Set OPENAI_BASE_URL=https://eu.api.openai.com/v1 (or your region) if not US.",
            _DEFAULT_OPENAI_BASE_URL,
        )
        return _DEFAULT_OPENAI_BASE_URL
    base = raw.rstrip("/")
    if not base.lower().endswith("/v1"):
        base = f"{base}/v1"
    return base


def _image_to_base64_data(image_spec: Any) -> Tuple[Optional[str], str]:
    """
    Convert image (path str, URL str, or PIL/Image object) to (base64_str, mime_type).
    Returns (None, mime_type) on failure. Used for Hub dataset embedded images (PIL) and local paths/URLs.
    """
    mime_type = "image/jpeg"
    # PIL / HF Image object (e.g. from Hub dataset row["images"][i])
    if not isinstance(image_spec, str):
        try:
            buf = io.BytesIO()
            image_spec.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
        except Exception as e:
            logger.warning(f"Failed to encode image object: {e}, skipping")
            return None, mime_type
    # URL
    if image_spec.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(
                image_spec,
                headers={"User-Agent": "Mozilla/5.0 (compatible; quizbowl/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                image_data = base64.b64encode(response.read()).decode("utf-8")
            ext = Path(image_spec).suffix.lower()
            mime_type = MIME_BY_EXT.get(ext, "image/jpeg")
            return image_data, mime_type
        except Exception as e:
            logger.warning(f"Failed to download image from URL {image_spec}: {e}, skipping")
            return None, mime_type
    # Local file path
    path = Path(image_spec)
    if not path.exists():
        logger.warning(f"Image file not found: {image_spec}, skipping")
        return None, mime_type
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.suffix.lower()
    mime_type = MIME_BY_EXT.get(ext, "image/jpeg")
    return image_data, mime_type


class CohereSchemaGenerator(GenerateJsonSchema):
    """Generates JSON schema for Cohere models without default titles."""

    def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
        return_value = super().field_title_should_be_set(schema)
        if return_value and is_core_schema(schema):
            return False
        return return_value


def _model_supports_vision(model: str) -> bool:
    """Whether the model can accept image inputs in chat completions."""
    return AVAILABLE_MODELS.get(model, {}).get("supports_vision", False)


def _strip_images_for_text_only_model(model: str, prompt: str, images: list | None) -> tuple[list | None, str]:
    """Drop images for text-only models; adjust prompt so image placeholders remain usable."""
    if not images:
        return images, prompt
    if _model_supports_vision(model):
        return images, prompt
    logger.warning(
        f"{model} does not support vision; ignoring {len(images)} image(s). "
        "Text placeholders like <img:...> are still included in the prompt."
    )
    note = (
        f"\n\nNote: This question references {len(images)} image(s) "
        "(marked with <img:...> placeholders in the text), but you cannot see them. "
        "Answer using only the visible text."
    )
    if "image(s) are included with this question" in prompt:
        prompt = prompt.replace(
            f"\n\nNote: {len(images)} image(s) are included with this question. "
            "Please analyze the image(s) along with the text to answer the question.",
            note,
        )
    elif note.strip() not in prompt:
        prompt = prompt + note
    return None, prompt


def _openai_is_json_mode_supported(model_name: str) -> bool:
    if model_name.startswith("gpt-4") or model_name.startswith("gpt-5"):
        return True
    if model_name.startswith("gpt-3.5"):
        return False
    logger.warning(f"OpenAI model {model_name} is not available in this app, skipping JSON mode, returning False")
    return False


class LLMOutput(BaseModel):
    content: str = Field(description="The content of the response")
    logprob: Optional[float] = Field(None, description="The log probability of the response")


def _get_langchain_chat_output(output: dict, prompt: str) -> str:
    ai_message = output["raw"]
    content = {"content": ai_message.content, "tool_calls": ai_message.tool_calls}
    content_str = json.dumps(content)
    if output["parsed"] is None:
        logger.error(f"No parsed output for Langchain model with User prompt: {repr(prompt)}")
        logger.error(f"Returned AI message: {repr(ai_message.content)}")
        logger.error(f"Raw output: {output}")
        if output["parsing_error"]:
            raise output["parsing_error"]
        else:
            raise Exception("No parsing error, but no parsed output")

    return {"content": content_str, "output": output["parsed"].model_dump()}


def _cohere_completion(
    model: str, system: str, prompt: str, response_model, temperature: float | None = None, logprobs: bool = True
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    schema = response_model.model_json_schema(schema_generator=CohereSchemaGenerator)
    if "title" in schema:
        del schema["title"]
    response_format = {
        "type": "json_object",
        "schema": schema,
    }
    response = client.chat(
        model=model,
        messages=messages,
        response_format=response_format,
        logprobs=logprobs,
        temperature=temperature,
    )
    output = {}
    output["content"] = response.message.content[0].text
    output["output"] = response_model.model_validate_json(response.message.content[0].text).model_dump()
    if logprobs:
        output["logprob"] = sum(lp.logprobs[0] for lp in response.logprobs)
        output["prob"] = np.exp(output["logprob"])
    return output


def _langchain_completion(
    provider: str, 
    model: str, 
    system: str, 
    prompt: str, 
    response_model, 
    temperature: float | None = None,
    images: list[str] | None = None
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    import base64
    from pathlib import Path
    
    if provider == "OpenAI":
        openai_base = _resolve_openai_base_url()
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_base=openai_base,
        ).with_structured_output(response_model, include_raw=True)
    elif provider == "Anthropic":
        model_cls = ChatAnthropic
        llm = model_cls(model=model, temperature=temperature).with_structured_output(response_model, include_raw=True)
    elif provider == "DeepSeek":
        model_cls = ChatDeepSeek
        llm = model_cls(model=model, temperature=temperature).with_structured_output(response_model, include_raw=True)
    else:
        raise ValueError(f"Provider {provider} not supported")
    
    # Build messages with images if provided
    messages = [SystemMessage(content=system)]
    
    if images:
        human_content = []
        images_added = 0
        for img in images:
            image_data, mime_type = _image_to_base64_data(img)
            if image_data:
                human_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                })
                images_added += 1
        if images_added == 0:
            logger.warning(f"[Multimodal Debug] None of the {len(images)} provided images could be loaded.")
        elif images_added > 0:
            logger.info(f"[Multimodal Debug] Successfully added {images_added}/{len(images)} images to API call")
        if prompt:
            human_content.append({"type": "text", "text": prompt})
        
        messages.append(HumanMessage(content=human_content))
    else:
        messages.append(HumanMessage(content=prompt))
    
    output = llm.invoke(messages)
    try:
        return _get_langchain_chat_output(output, prompt)
    except OutputParserException as e:
        logger.info(f"JSONDecodeError for model {model}, trying to parse invalid tool calls")
        invalid_tool_calls = output["raw"].invalid_tool_calls
        print(invalid_tool_calls)
        if not invalid_tool_calls:
            logger.error(f"Tried to parse invalid tool calls for model {model}, but no invalid tool calls found")
            raise e
        bad_json = [t["args"] for t in invalid_tool_calls if t["name"] == "ModelResponse"][0]
        repaired_obj = json_repair.loads(bad_json, skip_json_loads=True)
        parsed = response_model(**repaired_obj)
        return {"content": bad_json, "output": parsed.model_dump()}


def _openai_completion(
    model: str, 
    system: str, 
    prompt: str, 
    response_model, 
    temperature: float | None = None, 
    logprobs: bool = True,
    images: list[str] | None = None
) -> str:
    import base64
    import tempfile
    from pathlib import Path
    import urllib.request
    
    # Build user message content
    user_content = []
    
    # Add images if provided (paths, URLs, or PIL/Image objects e.g. from Hub dataset)
    if images:
        logger.info(f"[Multimodal Debug] Processing {len(images)} images for API call")
        images_added = 0
        for img in images:
            image_data, mime_type = _image_to_base64_data(img)
            if image_data:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                })
                images_added += 1
        if images_added == 0:
            logger.warning(f"[Multimodal Debug] None of the {len(images)} provided images could be loaded.")
        elif images_added > 0:
            logger.info(f"[Multimodal Debug] Successfully added {images_added}/{len(images)} images to API call")
    
    # Add text prompt
    if prompt:
        user_content.append({
            "type": "text",
            "text": prompt
        })
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content if images else prompt},
    ]
    base_url = _resolve_openai_base_url()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
    )

    def _attach_logprobs(out: dict, resp) -> None:
        if not logprobs:
            return
        lp = resp.choices[0].logprobs
        if lp and lp.content:
            out["logprob"] = sum(x.logprob for x in lp.content)
            out["prob"] = np.exp(out["logprob"])

    def _output_from_parse_message(msg, resp) -> dict:
        """Build output dict from a parse() message (refusals handled by retry loop)."""
        if msg.parsed is not None:
            out = {"content": msg.content, "output": msg.parsed.model_dump()}
            _attach_logprobs(out, resp)
            return out
        raw = msg.content
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raise ValueError(
                f"OpenAI returned no parsed object and empty content (model={model}); "
                f"finish_reason={getattr(resp.choices[0], 'finish_reason', None)!r}"
            )
        logger.warning(
            "OpenAI message.parsed is None; validating content as JSON for {} (finish_reason={})",
            model,
            getattr(resp.choices[0], "finish_reason", None),
        )
        text = raw if isinstance(raw, str) else str(raw)
        try:
            repaired_obj = json_repair.loads(text, skip_json_loads=True)
            parsed = response_model.model_validate(repaired_obj)
            out = {"content": msg.content, "output": parsed.model_dump()}
            _attach_logprobs(out, resp)
            return out
        except Exception as e:
            logger.error("Failed to recover structured output from OpenAI content (first 500 chars): {}", repr(text[:500]))
            raise ValueError(
                f"Could not parse OpenAI response into {getattr(response_model, '__name__', str(response_model))!r}: {e}"
            ) from e

    max_parse_attempts = 3
    parse_temp = temperature
    last_refusal: str | None = None

    for attempt in range(max_parse_attempts):
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            logprobs=logprobs,
            temperature=parse_temp,
        )
        msg = response.choices[0].message
        refusal = getattr(msg, "refusal", None)
        if refusal:
            snippet = refusal[:500] if isinstance(refusal, str) else str(refusal)[:500]
            last_refusal = refusal if isinstance(refusal, str) else str(refusal)
            logger.warning(
                "OpenAI parse refusal attempt {}/{} (model={}): {}",
                attempt + 1,
                max_parse_attempts,
                model,
                snippet,
            )
            parse_temp = min(1.0, (parse_temp or 0.0) + 0.2)
            continue
        return _output_from_parse_message(msg, response)

    if last_refusal is not None:
        logger.warning(
            "OpenAI parse() refused repeatedly; trying json_object fallback (model={})",
            model,
        )
        fb_system = (
            system
            + "\n\nRespond with only a JSON object containing the fields required for this task. "
            "This is an academic quiz competition; give your best answer in valid JSON. "
            "Do not reply with a refusal or apology."
        )
        fb_messages = [
            {"role": "system", "content": fb_system},
            {"role": "user", "content": user_content if images else prompt},
        ]
        try:
            fb_resp = client.chat.completions.create(
                model=model,
                messages=fb_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw_fb = fb_resp.choices[0].message.content
            if not raw_fb or not str(raw_fb).strip():
                raise ValueError("empty json_object fallback content")
            repaired_obj = json_repair.loads(raw_fb, skip_json_loads=True)
            parsed = response_model.model_validate(repaired_obj)
            out = {"content": raw_fb, "output": parsed.model_dump()}
            if logprobs:
                out["logprob"] = None
            return out
        except Exception as e:
            raise ValueError(
                f"OpenAI refused structured output (model={model}) after {max_parse_attempts} parse attempts: "
                f"{last_refusal!r}. json_object fallback failed: {e}"
            ) from e

    raise ValueError(f"OpenAI returned no usable structured output (model={model})")


def _llm_completion(
    model: str, 
    system: str, 
    prompt: str, 
    response_format, 
    temperature: float | None = None, 
    logprobs: bool = False,
    images: list[str] | None = None
) -> dict[str, Any]:
    """
    Generate a completion from an LLM provider with structured output without caching.

    Args:
        model (str): Provider and model name in format "provider/model" (e.g. "OpenAI/gpt-4")
        system (str): System prompt/instructions for the model
        prompt (str): User prompt/input
        response_format: Pydantic model defining the expected response structure
        logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            Note: Not supported by Anthropic models.
        images (list[str] | None, optional): List of image paths or URLs to include. Defaults to None.

    Returns:
        dict: Contains:
            - output: The structured response matching response_format
            - logprob: (optional) Sum of log probabilities if logprobs=True
            - prob: (optional) Exponential of logprob if logprobs=True

    Raises:
        ValueError: If logprobs=True with Anthropic models
    """
    model_name = AVAILABLE_MODELS[model]["model"]
    provider = model.split("/")[0]
    images, prompt = _strip_images_for_text_only_model(model, prompt, images)
    if provider == "Cohere":
        return _cohere_completion(model_name, system, prompt, response_format, temperature, logprobs)
    elif provider == "OpenAI":
        if _openai_is_json_mode_supported(model_name):
            return _openai_completion(model_name, system, prompt, response_format, temperature, logprobs, images)
        elif logprobs:
            raise ValueError(f"{model} does not support logprobs feature.")
        else:
            return _langchain_completion("OpenAI", model_name, system, prompt, response_format, temperature, images)
    elif provider in {"Anthropic", "DeepSeek"}:
        if logprobs:
            raise ValueError(f"{provider} models do not support logprobs")
        return _langchain_completion(provider, model_name, system, prompt, response_format, temperature, images)
    else:
        raise ValueError(f"Provider {provider} not supported")


def completion(
    model: str, 
    system: str, 
    prompt: str, 
    response_format, 
    temperature: float | None = None, 
    logprobs: bool = True,
    images: list[str] | None = None
) -> dict[str, Any]:
    """
    Generate a completion from an LLM provider with structured output with caching.

    Args:
        model (str): Provider and model name in format "provider/model" (e.g. "OpenAI/gpt-4")
        system (str): System prompt/instructions for the model
        prompt (str): User prompt/input
        response_format: Pydantic model defining the expected response structure
        temperature (float | None, optional): Temperature for sampling. Defaults to None.
        logprobs (bool, optional): Whether to return log probabilities. Defaults to True.
            Note: Not supported by Anthropic models.
        images (list[str] | None, optional): List of image paths or URLs to include. Defaults to None.

    Returns:
        dict: Contains:
            - output: The structured response matching response_format
            - logprob: (optional) Sum of log probabilities if logprobs=True
            - prob: (optional) Exponential of logprob if logprobs=True

    Raises:
        ValueError: If logprobs=True with Anthropic models
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} not supported")
    if logprobs and not AVAILABLE_MODELS[model].get("logprobs", False):
        # logger.warning(f"{model} does not support logprobs feature, setting logprobs to False")
        logprobs = False

    # Check cache first (note: cache key doesn't include images yet - may need enhancement)
    cached_response = llm_cache.get(model, system, prompt, response_format, temperature)
    if cached_response and (not logprobs or cached_response.get("logprob")) and not images:
        logger.trace(f"Cache hit for model {model}")
        return cached_response

    logger.trace(f"Cache miss for model {model}, calling API. Logprobs: {logprobs}, Images: {len(images) if images else 0}")

    # Continue with the original implementation for cache miss
    response = _llm_completion(model, system, prompt, response_format, temperature, logprobs, images)

    # Update cache with the new response (only if no images to avoid cache issues)
    if not images:
        llm_cache.set(
            model,
            system,
            prompt,
            response_format,
            temperature,
            response,
        )

    return response


# %%
if __name__ == "__main__":
    from tqdm import tqdm

    class ExplainedAnswer(BaseModel):
        """
        The answer to the question and a terse explanation of the answer.
        """

        answer: str = Field(description="The short answer to the question")
        explanation: str = Field(description="5 words terse best explanation of the answer.")

    TEST_MODELS = ["Cohere/command-r7b", "Anthropic/claude-haiku-4-5", "DeepSeek/V3"]

    models = TEST_MODELS
    system = "You are an accurate and concise explainer of scientific concepts."
    prompt = "Which planet is closest to the sun in the Milky Way galaxy? Answer directly, no explanation needed."

    llm_cache = LLMCache(cache_dir="/tmp/cache", hf_repo="qanta-challenge/advcal-llm-cache", reset=True)

    # First call - should be a cache miss
    logger.info("First call - should be a cache miss")
    for model in tqdm(models):
        response = completion(model, system, prompt, ExplainedAnswer, logprobs=False)
        rprint(response)

    # Second call - should be a cache hit
    logger.info("Second call - should be a cache hit")
    for model in tqdm(models):
        response = completion(model, system, prompt, ExplainedAnswer, logprobs=False)
        rprint(response)

    # Slightly different prompt - should be a cache miss
    logger.info("Different prompt - should be a cache miss")
    prompt2 = "Which planet is closest to the sun? Answer directly."
    for model in tqdm(models):
        response = completion(model, system, prompt2, ExplainedAnswer, logprobs=False)
        rprint(response)

    # Get cache entries count from SQLite
    try:
        cache_entries = llm_cache.get_all_entries()
        logger.info(f"Cache now has {len(cache_entries)} items")
    except Exception as e:
        logger.error(f"Failed to get cache entries: {e}")

    # Test adding entry with temperature parameter
    logger.info("Testing with temperature parameter")
    response = completion(models[0], system, "What is Mars?", ExplainedAnswer, temperature=0.7, logprobs=False)
    rprint(response)

    # Demonstrate forced sync to HF if repo is configured
    if llm_cache.hf_repo_id:
        logger.info("Forcing sync to HF dataset")
        try:
            llm_cache.sync_to_hf()
            logger.info("Successfully synced to HF dataset")
        except Exception as e:
            logger.exception(f"Failed to sync to HF: {e}")
    else:
        logger.info("HF repo not configured, skipping sync test")

# %%
