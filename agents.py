from __future__ import annotations

import importlib
import json
import os
import random
import time
from review_schema import parse_json_object, validate_role_output


VALID_PROVIDERS = {"cmu", "openai", "gemini", "claude", "deepseek", "qwen", "openrouter"}
DEFAULT_MODELS = {
    "cmu": "gpt-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
    "claude": "claude-3-5-sonnet-20240620",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen3.7-plus",
    "openrouter": "google/gemini-2.5-flash",
}

OPENAI_COMPATIBLE_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
OPENAI_COMPATIBLE_ENV_BASE_URL = {
    "deepseek": "DEEPSEEK_BASE_URL",
    "qwen": "QWEN_BASE_URL",
    "openrouter": "OPENAI_API_BASE",
}
# Keep default OpenRouter max_tokens moderate (4096) for standard chat models so
# OpenRouter does not reserve 25,600 completion tokens per concurrent request
# against the account's in-flight credit budget (which triggers HTTP 402
# in_flight_budget_exhausted). Reasoning/thinking models or explicit
# OPENROUTER_MAX_TOKENS overrides can request a larger cap.
OPENAI_COMPATIBLE_MAX_TOKENS = {
    "openrouter": 4096,
}
OPENROUTER_REASONING_MAX_TOKENS = 16384
OPENAI_COMPATIBLE_PARSE_RETRY_DELAYS = (1.0, 2.0, 4.0)
REQUIRED_JSON_RETRY_DELAYS = (1.0, 2.0, 4.0)


def _resolve_openrouter_max_tokens(model: str) -> int:
    env_override = os.environ.get("OPENROUTER_MAX_TOKENS", "").strip()
    if env_override:
        try:
            value = int(env_override)
            if value > 0:
                return value
        except ValueError:
            pass
    model_lower = (model or "").lower()
    if any(tag in model_lower for tag in ("thinking", "deepseek-r1", "/r1", "qwq", "/o1", "/o3", "-r1")):
        return OPENROUTER_REASONING_MAX_TOKENS
    return OPENAI_COMPATIBLE_MAX_TOKENS["openrouter"]


def _is_transient_provider_error(exc: Exception, attempt: int) -> tuple[bool, float]:
    status_code = getattr(exc, "status_code", None)
    text = str(exc)
    text_lower = text.lower()
    base_delay = REQUIRED_JSON_RETRY_DELAYS[min(attempt, len(REQUIRED_JSON_RETRY_DELAYS) - 1)]

    is_in_flight_402 = (
        status_code == 402
        and ("in_flight_budget_exhausted" in text_lower or "in-flight" in text_lower)
    )
    transient = (
        isinstance(exc, (TimeoutError, ConnectionError))
        or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}
        or status_code in {408, 429, 500, 502, 503, 504}
        or is_in_flight_402
    )
    if not transient:
        return False, 0.0

    retry_after = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers and hasattr(headers, "get"):
        raw_retry = headers.get("Retry-After") or headers.get("retry-after")
        if raw_retry is not None:
            try:
                retry_after = float(raw_retry)
            except (TypeError, ValueError):
                retry_after = None
    if retry_after is None and is_in_flight_402:
        retry_after = 15.0 * (attempt + 1)
    delay = min(max(retry_after, base_delay), 120.0) if retry_after is not None else base_delay
    return True, delay


def _parse_required_json_object(text: str) -> dict:
    """Parse a model reply that is required to be a JSON object."""
    return parse_json_object(text)

def validate_api_key_for_provider(provider: str, api_key: str) -> str:
    provider = (provider or "cmu").lower()
    api_key = (api_key or "").strip()
    if not api_key:
        return "API key is required."
    if provider == "cmu" and not api_key.startswith("sk-"):
        return (
            "CMU AI Gateway expects a gateway key that starts with 'sk-'. "
            "Select the matching provider for this key, or paste the CMU gateway key."
        )
    if provider == "openai" and not api_key.startswith("sk-"):
        return "OpenAI API keys should start with 'sk-'. Select the matching provider for this key."
    if provider == "deepseek" and not api_key.startswith("sk-"):
        return "DeepSeek API keys should start with 'sk-'. Select the matching provider for this key."
    if provider == "qwen" and not api_key.startswith("sk-"):
        return "Qwen / Model Studio API keys should start with 'sk-'. Select the matching provider for this key."
    if provider == "openrouter" and not api_key.startswith("sk-"):
        return "OpenRouter API keys should start with 'sk-'. Select the matching provider for this key."
    return ""


def format_llm_error(provider: str, exc: Exception) -> str:
    provider = (provider or "cmu").lower()
    text = str(exc)
    looks_like_auth = (
        "401" in text
        or "auth" in text.lower()
        or "api key" in text.lower()
        or "virtual key" in text.lower()
    )
    if looks_like_auth and provider == "cmu":
        return (
            "CMU AI Gateway authentication failed. Use the CMU gateway key for the selected "
            "provider; it should start with 'sk-'."
        )
    if looks_like_auth and provider == "openai":
        return "OpenAI authentication failed. Check that the selected provider and API key match."
    if looks_like_auth and provider == "gemini":
        return "Gemini authentication failed. Check that the selected provider and API key match."
    if looks_like_auth and provider == "claude":
        return "Claude authentication failed. Check that the selected provider and API key match."
    if looks_like_auth and provider == "deepseek":
        return "DeepSeek authentication failed. Check your DeepSeek API key."
    if looks_like_auth and provider == "qwen":
        return "Qwen authentication failed. Check that the API key and Qwen region endpoint match."
    if looks_like_auth and provider == "openrouter":
        return "OpenRouter authentication failed. Check that the selected provider and API key match."
    return text


def _get_api_key(provider: str = "cmu") -> str:
    env_keys = {
        "cmu": ("API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "qwen": ("DASHSCOPE_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    }
    candidates = env_keys.get(provider, ("API_KEY",))
    try:
        from google.colab import userdata
        for env_name in candidates:
            key = userdata.get(env_name)
            if key:
                return key
    except Exception:
        pass
    for env_name in candidates:
        if os.environ.get(env_name):
            return os.environ[env_name]
    return os.environ[candidates[0]]


def _as_alternating_chat(messages: list[dict]) -> list[dict]:
    chat = []
    for msg in messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if chat and chat[-1]["role"] == role:
            chat[-1]["content"] += "\n\n" + content
        else:
            chat.append({"role": role, "content": content})
    return chat


class CMUGatewayClient:
    def __init__(self, api_key: str, model: str):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "CMU support requires the openai package. "
                "Install requirements.txt and try again."
            ) from exc
        self.model = model or DEFAULT_MODELS["cmu"]
        self.client = openai.OpenAI(
            api_key=api_key or _get_api_key("cmu"),
            base_url="https://ai-gateway.andrew.cmu.edu",
        )

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
        )
        return response.choices[0].message.content.strip()


class OpenAIChatGPTClient:
    def __init__(self, api_key: str, model: str):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires the openai package. "
                "Install requirements.txt and try again."
            ) from exc
        self.model = model or DEFAULT_MODELS["openai"]
        self.client = openai.OpenAI(api_key=api_key or _get_api_key("openai"))

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
        )
        return response.choices[0].message.content.strip()


class OpenAICompatibleClient:
    """Chat-completions client for providers exposing an OpenAI-compatible API."""

    def __init__(self, provider: str, api_key: str, model: str):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                f"{provider.title()} support requires the openai package. "
                "Install requirements.txt and try again."
            ) from exc
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.base_url = os.environ.get(
            OPENAI_COMPATIBLE_ENV_BASE_URL[provider], OPENAI_COMPATIBLE_BASE_URLS[provider]
        )
        self.client = openai.OpenAI(
            api_key=api_key or _get_api_key(provider),
            base_url=self.base_url,
        )
        if provider == "openrouter":
            self.max_tokens = _resolve_openrouter_max_tokens(self.model)
        else:
            self.max_tokens = OPENAI_COMPATIBLE_MAX_TOKENS.get(provider)

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        kwargs = {"max_tokens": self.max_tokens} if self.max_tokens is not None else {}
        request_messages = [{"role": "system", "content": system_prompt}, *messages]
        max_attempts = len(OPENAI_COMPATIBLE_PARSE_RETRY_DELAYS) + 1
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    **kwargs,
                )
                break
            except json.JSONDecodeError:
                if attempt + 1 >= max_attempts:
                    raise
                base_delay = OPENAI_COMPATIBLE_PARSE_RETRY_DELAYS[attempt]
                delay = base_delay + random.uniform(0.0, base_delay * 0.25)
                print(
                    f"[{self.provider}] Malformed API response; retrying request "
                    f"{attempt + 2}/{max_attempts} in {delay:.1f}s..."
                )
                time.sleep(delay)
        return (response.choices[0].message.content or "").strip()


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the google-genai package. "
                "Install requirements.txt and try again."
            ) from exc
        self.model = model or DEFAULT_MODELS["gemini"]
        self.client = genai.Client(api_key=api_key or _get_api_key("gemini"))

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        contents = [
            {
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}],
            }
            for msg in _as_alternating_chat(messages)
        ]
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"system_instruction": system_prompt},
        )
        return (getattr(response, "text", "") or "").strip()


class ClaudeClient:
    def __init__(self, api_key: str, model: str):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude support requires the anthropic package. "
                "Install requirements.txt and try again."
            ) from exc
        self.model = model or DEFAULT_MODELS["claude"]
        self.client = anthropic.Anthropic(api_key=api_key or _get_api_key("claude"))

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=_as_alternating_chat(messages),
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()


def create_llm_client(provider: str, api_key: str, model: str):
    if provider == "cmu":
        return CMUGatewayClient(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIChatGPTClient(api_key=api_key, model=model)
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=model)
    if provider == "claude":
        return ClaudeClient(api_key=api_key, model=model)
    if provider in OPENAI_COMPATIBLE_BASE_URLS:
        return OpenAICompatibleClient(provider=provider, api_key=api_key, model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")


_PROMPT_MAP = {
    "reviewer_a":        ("prompts.reviewer_a",        "reviewer_a"),
    "reviewer_b":        ("prompts.reviewer_b",        "reviewer_b"),
    "reviewer_c":        ("prompts.reviewer_c",        "reviewer_c"),
    "reviewer_nopersona":("prompts.reviewer_nopersona","reviewer_nopersona"),
    "author":            ("prompts.author",         "author"),
    "ai_detector":       ("prompts.ai_detector",    "ai_detector"),
    "reviewer_iteration":("prompts.reviewer_iter",  "reviewer_iteration"),
    "conf_rec":          ("prompts.conf_rec",       "Conference_Recommender"),
    "related_work_analyst": ("prompts.related_work_analyst", "related_work_analyst"),
}


def _load_prompt(key: str) -> str:
    module_name, var_name = _PROMPT_MAP[key]
    module = importlib.import_module(module_name)
    return getattr(module, var_name)


def _inject_topic(persona: str, topic: str) -> str:
    """Prepend the paper topic to the agent persona so every agent is topic-aware."""
    if not topic:
        return persona
    header = f"###Paper Topic###\nThe paper belongs to the following research area: {topic}\n\n"
    return header + persona


class Agent:
    """
    An LLM model with different personas.
    Initialized to a general goal: review paper / author of a paper.

    Args:
        persona: System-level persona string.
        paper:   Full paper text.
        topic:   Research area selected by the author (e.g. "NLP"). Injected
                 into the persona so the agent applies topic-aware judgement.
        model:   LLM model name.
    """

    name = "Agent"
    output_role = "generic"

    def __init__(self, persona: str, paper: str, topic: str = "", model: str = "",
                 api_key: str = "", provider: str = "cmu"):
        print(f"[{self.name}] Initializing...")
        provider = (provider or "cmu").lower()
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.topic   = topic
        self.persona = _inject_topic(persona, topic)
        self.paper   = paper
        self.provider = provider
        self.model   = model or DEFAULT_MODELS[provider]
        self.client  = create_llm_client(provider=provider, api_key=api_key, model=self.model)
        self.messages = [
            {"role": "user",      "content": f"Here is the paper you will be working with:\n\n{paper}"}
        ]
        self.last_call_result = None
        print(f"[{self.name}] Ready.")

    def call(self, user_message: str, *, required_json: bool = True) -> str | None:
        """Send a message and return the agent's reply, maintaining conversation history."""
        print(f"[{self.name}] Getting response...")
        self.messages.append({"role": "user", "content": user_message})
        request_messages = [dict(message) for message in self.messages]
        diagnostic = {"role": self.output_role, "status": "running", "attempts": []}
        self.last_call_result = diagnostic
        attempts = len(REQUIRED_JSON_RETRY_DELAYS) + 1 if required_json else 1
        for attempt in range(attempts):
            try:
                reply = self.client.complete(self.persona, [dict(message) for message in request_messages])
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                attempt_record = {
                    "attempt": attempt + 1,
                    "status": "provider_error",
                    "error_type": type(exc).__name__,
                }
                if status_code is not None:
                    attempt_record["status_code"] = status_code
                diagnostic["attempts"].append(attempt_record)
                transient, delay = _is_transient_provider_error(exc, attempt)
                if transient and attempt + 1 < attempts:
                    time.sleep(delay)
                    continue
                diagnostic["status"] = "failed"
                raise
            if not required_json:
                self.messages.append({"role": "assistant", "content": reply})
                diagnostic["attempts"].append({"attempt": attempt + 1, "status": "valid"})
                diagnostic["status"] = "complete"
                print(f"[{self.name}] Done.\n")
                return reply
            try:
                parsed = _parse_required_json_object(reply)
                errors = validate_role_output(parsed, self.output_role, getattr(self, "expected_reviewer", None))
                extra_validator = getattr(self, "extra_output_validator", None)
                if not errors and extra_validator is not None:
                    errors = extra_validator(parsed)
                if errors:
                    raise ValueError("; ".join(errors))
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
                diagnostic["attempts"].append({"attempt": attempt + 1, "status": "invalid_output", "errors": [str(exc)], "raw_reply": reply})
                if attempt + 1 >= attempts:
                    diagnostic["status"] = "failed"
                    print(f"[{self.name}] Invalid JSON return discarded after {attempts} attempts.\n")
                    return None
                delay = REQUIRED_JSON_RETRY_DELAYS[attempt]
                print(
                    f"[{self.name}] Invalid JSON return discarded; retrying the same prompt "
                    f"{attempt + 2}/{attempts} in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue

            # Invalid attempts are deliberately absent from history. Only a valid
            # structured reply becomes part of the conversation.
            self.messages.append({"role": "assistant", "content": reply})
            diagnostic["attempts"].append({"attempt": attempt + 1, "status": "valid"})
            diagnostic["status"] = "complete"
            print(f"[{self.name}] Done.\n")
            return reply

        return None


class Reviewer(Agent):
    """
    An LLM agent with the persona of an academic paper reviewer.
    reviewer_type: "reviewer_a" (novelty-focused), "reviewer_b" (rigor-focused),
                   "reviewer_c" (practicality-focused), or "reviewer_nopersona"
    """

    output_role = "reviewer"

    def __init__(self, paper: str, reviewer_type: str = "reviewer_a",
                 topic: str = "", model: str = "", api_key: str = "",
                 provider: str = "cmu"):
        _label = {
            "reviewer_a":        "Novelty",
            "reviewer_b":        "Rigor",
            "reviewer_c":        "Practical",
            "reviewer_nopersona":"Neutral",
        }
        self.name = f"Reviewer ({_label.get(reviewer_type, reviewer_type)})"
        persona = _load_prompt(reviewer_type)
        super().__init__(
            persona=persona, paper=paper, topic=topic, model=model,
            api_key=api_key, provider=provider,
        )


class Author(Agent):
    """An LLM agent with the persona of the paper's author."""

    name = "Author"
    output_role = "author"

    def __init__(self, paper: str, topic: str = "", model: str = "",
                 api_key: str = "", provider: str = "cmu"):
        persona = _load_prompt("author")
        super().__init__(
            persona=persona, paper=paper, topic=topic, model=model,
            api_key=api_key, provider=provider,
        )


class AIDetector(Agent):
    """An LLM agent that evaluates human-like conference-review writing style."""

    name = "AI Detector (Review Style)"

    def __init__(self, paper: str, topic: str = "", model: str = "",
                 api_key: str = "", provider: str = "cmu"):
        persona = _load_prompt("ai_detector")
        super().__init__(
            persona=persona, paper=paper, topic=topic, model=model,
            api_key=api_key, provider=provider,
        )


class ConferenceRecommender(Agent):
    """
    An LLM agent that recommends the best-fit ML conference (ICML / NeurIPS / ICLR)
    given the paper, its topic, and accumulated reviewer scores.
    """

    name = "Conference Recommender"
    output_role = "conference"

    def __init__(self, paper: str, topic: str = "", model: str = "",
                 api_key: str = "", provider: str = "cmu"):
        persona = _load_prompt("conf_rec")
        super().__init__(
            persona=persona, paper=paper, topic=topic, model=model,
            api_key=api_key, provider=provider,
        )


class RelatedWorkAnalyst(Agent):
    """
    Compares the submission's claimed contributions with retrieved prior work (RAG improvement option 2).

    It is given a short context (abstract, claimed contributions and the labeled retrieved works)
    instead of the full paper, so the comparison does not get lost in a long prompt.
    """

    name = "Related-Work Analyst"
    output_role = "rw_analyst"

    def __init__(self, analyst_input: str, model: str = "", api_key: str = "", provider: str = "cmu"):
        persona = _load_prompt("related_work_analyst")
        super().__init__(
            persona=persona, paper=analyst_input, topic="", model=model,
            api_key=api_key, provider=provider,
        )
