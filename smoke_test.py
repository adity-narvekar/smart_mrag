from dotenv import load_dotenv
load_dotenv()

# smoke_test_providers.py
import os
from smart_mrag.llm_providers import get_llm_provider
from smart_mrag.embeddings_providers import get_embedding_provider

# --- Anthropic LLM ---
claude = get_llm_provider("claude-sonnet-4-5", os.getenv("ANTHROPIC_API_KEY"))
print("Claude says:", claude.generate("Say 'hello' and nothing else."))

# --- Gemini LLM ---
gemini = get_llm_provider("gemini-3.6-flash", os.getenv("GOOGLE_API_KEY"))
print("Gemini says:", gemini.generate("Say 'hello' and nothing else."))

# --- Gemini embeddings ---
gemini_embed = get_embedding_provider("gemini-embedding-001", os.getenv("GOOGLE_API_KEY"))
vec = gemini_embed.embed("test sentence")
print("Gemini embedding length:", len(vec))

# --- OpenAI embeddings (used for Claude's retrieval step) ---
openai_embed = get_embedding_provider("text-embedding-ada-002", os.getenv("OPENAI_API_KEY"))
vec2 = openai_embed.embed("test sentence")
print("OpenAI embedding length:", len(vec2))