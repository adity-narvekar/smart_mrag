from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key, model_name, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(input=text, model=self.model_name)
        return resp.data[0].embedding


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key, model_name="gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return resp.embeddings[0].values


def get_embedding_provider(embedding_model: str, api_key: str, base_url=None) -> BaseEmbeddingProvider:
    name = embedding_model.lower()
    if "gecko" in name or name.startswith("models/text-embedding") or name.startswith("gemini-embedding"):
        return GoogleEmbeddingProvider(api_key, embedding_model)
    if "text-embedding" in name:
        return OpenAIEmbeddingProvider(api_key, embedding_model, base_url)
    raise ValueError(f"Unsupported embedding model: {embedding_model}")