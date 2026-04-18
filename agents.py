import importlib
import openai


def _get_api_key() -> str:
    try:
        from google.colab import userdata
        return userdata.get("API_KEY")
    except Exception:
        import os
        return os.environ["API_KEY"]


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

    def __init__(self, persona: str, paper: str, topic: str = "", model: str = "gpt-5", api_key: str = ""):
        print(f"[{self.name}] Initializing...")
        self.topic   = topic
        self.persona = _inject_topic(persona, topic)
        self.paper   = paper
        self.model   = model
        self.client  = openai.OpenAI(
            api_key=api_key or _get_api_key(),
            base_url="https://ai-gateway.andrew.cmu.edu"
        )
        self.messages = [
            {"role": "developer", "content": self.persona},
            {"role": "user",      "content": f"Here is the paper you will be working with:\n\n{paper}"}
        ]
        print(f"[{self.name}] Ready.")

    def call(self, user_message: str) -> str:
        """Send a message and return the agent's reply, maintaining conversation history."""
        print(f"[{self.name}] Getting response...")
        self.messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        reply = response.choices[0].message.content.strip()
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
                 topic: str = "", model: str = "gpt-5", api_key: str = ""):
        _label = {
            "reviewer_a":        "Novelty",
            "reviewer_b":        "Rigor",
            "reviewer_c":        "Practical",
            "reviewer_nopersona":"Neutral",
        }
        self.name = f"Reviewer ({_label.get(reviewer_type, reviewer_type)})"
        persona = _load_prompt(reviewer_type)
        super().__init__(persona=persona, paper=paper, topic=topic, model=model, api_key=api_key)


class Author(Agent):
    """An LLM agent with the persona of the paper's author."""

    name = "Author"

    def __init__(self, paper: str, topic: str = "", model: str = "gpt-5", api_key: str = ""):
        persona = _load_prompt("author")
        super().__init__(persona=persona, paper=paper, topic=topic, model=model, api_key=api_key)


class AIDetector(Agent):
    """An LLM agent that detects whether writing is AI-generated."""

    name = "AI Detector"

    def __init__(self, paper: str, topic: str = "", model: str = "gpt-5", api_key: str = ""):
        persona = _load_prompt("ai_detector")
        super().__init__(persona=persona, paper=paper, topic=topic, model=model, api_key=api_key)


class ConferenceRecommender(Agent):
    """
    An LLM agent that recommends the best-fit ML conference (ICML / NeurIPS / ICLR)
    given the paper, its topic, and accumulated reviewer scores.
    """

    name = "Conference Recommender"

    def __init__(self, paper: str, topic: str = "", model: str = "gpt-5", api_key: str = ""):
        persona = _load_prompt("conf_rec")
        super().__init__(persona=persona, paper=paper, topic=topic, model=model, api_key=api_key)
