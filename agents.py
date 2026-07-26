import importlib
import os


VALID_PROVIDERS = {"cmu", "openai", "gemini", "claude", "deepseek", "qwen", "openrouter"}
DEFAULT_MODELS = {
    "cmu": "gpt-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
    "claude": "claude-3-5-sonnet-20240620",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen3.7-plus",
    "openrouter": "google/gemini-3.5-flash",
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
# OpenRouter fans out to third-party "thinking" models that spend part of
# max_tokens on hidden reasoning tokens before ever producing visible
# content; cap it well above the reasoning budget so replies aren't empty.
OPENAI_COMPATIBLE_MAX_TOKENS = {
    "openrouter": 25600,
}


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
        "cmu": "API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "openrouter": "OPENAI_API_KEY",
    }
    try:
        from google.colab import userdata
        key = userdata.get(env_keys.get(provider, "API_KEY"))
        if key:
            return key
    except Exception:
        pass
    return os.environ[env_keys.get(provider, "API_KEY")]


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
        self.max_tokens = OPENAI_COMPATIBLE_MAX_TOKENS.get(provider)

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        kwargs = {"max_tokens": self.max_tokens} if self.max_tokens is not None else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            **kwargs,
        )
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
        print(f"[{self.name}] Ready.")

    def call(self, user_message: str) -> str:
        """Send a message and return the agent's reply, maintaining conversation history."""
        print(f"[{self.name}] Getting response...")
        self.messages.append({"role": "user", "content": user_message})
        reply = self.client.complete(self.persona, self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        print(f"[{self.name}] Done.\n")
        return reply


class Reviewer(Agent):
    """
    An LLM agent with the persona of an academic paper reviewer.
    reviewer_type: "reviewer_a" (novelty-focused), "reviewer_b" (rigor-focused),
                   "reviewer_c" (practicality-focused), or "reviewer_nopersona"
    """

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

    def __init__(self, paper: str, topic: str = "", model: str = "",
                 api_key: str = "", provider: str = "cmu"):
        persona = _load_prompt("conf_rec")
        super().__init__(
            persona=persona, paper=paper, topic=topic, model=model,
            api_key=api_key, provider=provider,
        )
