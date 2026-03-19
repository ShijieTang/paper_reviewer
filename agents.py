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
    "reviewer_a":        ("prompts.reviewer_a",    "reviewer_a"),
    "reviewer_b":        ("prompts.reviewer_b",    "reviewer_b"),
    "author":            ("prompts.author",         "author"),
    "ai_detector":       ("prompts.ai_detector",    "ai_detector"),
    "reviewer_iteration":("prompts.reviewer_iter",  "reviewer_iteration"),
}


def _load_prompt(key: str) -> str:
    module_name, var_name = _PROMPT_MAP[key]
    module = importlib.import_module(module_name)
    return getattr(module, var_name)


class Agent:
    '''
    An LLM model with different personas. Initialized to a general goal: review paper/author of a paper
    '''

    name = "Agent"

    def __init__(self, persona: str, paper: str, model: str = "gpt-5"):
        print(f"[{self.name}] Initializing...")
        self.persona = persona
        self.paper = paper
        self.model = model
        self.client = openai.OpenAI(
            api_key=_get_api_key(),
            base_url="https://ai-gateway.andrew.cmu.edu"
        )
        self.messages = [
            {"role": "developer", "content": persona},
            {"role": "user", "content": f"Here is the paper you will be working with:\n\n{paper}"}
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
    '''
    An LLM agent with the persona of an academic paper reviewer.
    reviewer_type: "reviewer_a" (novelty-focused) or "reviewer_b" (rigor-focused)
    '''

    def __init__(self, paper: str, reviewer_type: str = "reviewer_a", model: str = "gpt-5"):
        self.name = f"Reviewer ({reviewer_type})"
        persona = _load_prompt(reviewer_type)
        super().__init__(persona=persona, paper=paper, model=model)


class Author(Agent):
    '''
    An LLM agent with the persona of the paper's author.
    '''

    name = "Author"

    def __init__(self, paper: str, model: str = "gpt-5"):
        persona = _load_prompt("author")
        super().__init__(persona=persona, paper=paper, model=model)


class AIDetector(Agent):
    '''
    An LLM agent with the persona of an AI detector.
    '''

    name = "AIDetector"

    def __init__(self, paper: str, model: str = "gpt-5"):
        persona = _load_prompt("ai_detector")
        super().__init__(persona=persona, paper=paper, model=model)
