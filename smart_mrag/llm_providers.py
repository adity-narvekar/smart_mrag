from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Common interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens=4096):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return resp.text


def get_llm_provider(model_name: str, api_key: str, base_url=None) -> BaseLLMProvider:
    """Factory: pick the right provider class from the model name."""
    name = model_name.lower()
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return OpenAIProvider(api_key, model_name, base_url)
    if name.startswith("claude"):
        return AnthropicProvider(api_key, model_name)
    if name.startswith("gemini"):
        return GeminiProvider(api_key, model_name)
    raise ValueError(
        f"Unsupported model '{model_name}'. Must start with gpt-, claude-, or gemini-."
    )