from pydantic import BaseModel, validator
from typing import Optional
import re

class ModelConfig(BaseModel):
    """Configuration for the RAG model, OpenAI only."""
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    openai_endpoint: Optional[str] = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = None
    similarity_threshold: float = 0.7
    top_k: int = 5

    @validator('openai_endpoint')
    def validate_endpoints(cls, v):
        if v is None:
            return v
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if not url_pattern.match(v):
            raise ValueError(f"Invalid endpoint URL: {v}")
        return v 