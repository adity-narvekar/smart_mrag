import os
from typing import Optional, List
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .llm_providers import get_llm_provider
from .embeddings_providers import get_embedding_provider
import fitz
from PIL import Image
import base64
import io
from .utils import ModelConfig
from tqdm import tqdm

__version__ = "0.1.4"

# All models
RECOMMENDED_MODELS = {
    "openai": { "llm_models": [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo-preview",
                "gpt-4-vision-preview",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4o-turbo"
            ],
            "embedding_models": [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-3-large-256"
            ],
            "requires": ["openai_api_key"] },
    "anthropic": {
        "llm_models": [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
        ],
        "embedding_models": [],  # Anthropic has no embedding API
        "requires": ["anthropic_api_key"]
    },
    "google": {
        "llm_models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        "embedding_models": [
            "models/text-embedding-004"
        ],
        "requires": ["google_api_key"]
    }
}
def get_recommended_models():
    return RECOMMENDED_MODELS

def get_required_api_keys(llm_model, embedding_model):
    required_keys = set()
    if any(m in llm_model for m in ["gpt-3.5", "gpt-4", "o1", "o3"]):
        required_keys.add("openai_api_key")
    if llm_model.startswith("claude"):
        required_keys.add("anthropic_api_key")
    if llm_model.startswith("gemini"):
        required_keys.add("google_api_key")
    if "text-embedding" in embedding_model and "gecko" not in embedding_model:
        required_keys.add("openai_api_key")
    if "gecko" in embedding_model or embedding_model.startswith("models/text-embedding"):
        required_keys.add("google_api_key")
    return list(required_keys)

__all__ = ["SmartMRAG", "ModelConfig", "get_recommended_models", "get_required_api_keys"]

class SmartMRAG:
    DEFAULT_MODELS = {
        "gpt-4o": {
            "embedding_model": "text-embedding-ada-002",
            "provider": "openai"
        },
        "gpt-4": {
            "embedding_model": "text-embedding-ada-002",
            "provider": "openai"
        },
        "gpt-4-turbo": {
            "embedding_model": "text-embedding-ada-002",
            "provider": "openai"
        },
        "gpt-4-vision": {
            "embedding_model": "text-embedding-ada-002",
            "provider": "openai"
        },
        "gpt-3.5-turbo": {
            "embedding_model": "text-embedding-ada-002",
            "provider": "openai"
        },
        "claude-sonnet-4-5": {
            "embedding_model": "text-embedding-ada-002",  # still OpenAI, since Anthropic has none
            "provider": "anthropic"
        },
        "gemini-2.5-flash": {
            "embedding_model": "models/text-embedding-004",
            "provider": "google"
},
    }
    @staticmethod
    def _infer_provider(model_name: str) -> str:
        name = model_name.lower()
        if name.startswith("claude"):
            return "anthropic"
        if name.startswith("gemini"):
            return "google"
        return "openai"

    def _resolve_embedding_key(self, embedding_model, embedding_api_key):
        name = embedding_model.lower()
        if "gecko" in name or name.startswith("models/text-embedding") or name.startswith("gemini-embedding"):
            return embedding_api_key or os.getenv("GOOGLE_API_KEY") or self.api_key
        return embedding_api_key or os.getenv("OPENAI_API_KEY") or self.api_key
    def __init__(
        self,
        file_path: str,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        openai_endpoint: Optional[str] = None
    ):
        self.file_path = file_path
        self.model_name = model_name
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        # NEW: figure out which vendor this model belongs to
        self.provider = self._infer_provider(model_name)
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY", 
        }
        env_var = env_var_map[self.provider]
        self.api_key = api_key or os.getenv(env_var)
        if not self.api_key:
            raise ValueError(f"API key is required. Please provide it or set {env_var} environment variable")
        
        if embedding_model:
            if model_name in self.DEFAULT_MODELS and embedding_model != self.DEFAULT_MODELS[model_name]["embedding_model"]:
                if not embedding_api_key:
                    raise ValueError(f"Embedding API key is required when using custom embedding model: {embedding_model}")
                self.embedding_api_key = embedding_api_key
            else:
                self.embedding_api_key = self._resolve_embedding_key(embedding_model, embedding_api_key)
        else:
            if model_name in self.DEFAULT_MODELS:
                embedding_model = self.DEFAULT_MODELS[model_name]["embedding_model"]
            else:
                raise ValueError("Embedding model is required when using a non-default model")
            self.embedding_api_key = self._resolve_embedding_key(embedding_model, embedding_api_key)

        self.embedding_model = embedding_model

            
        self.llm_provider = get_llm_provider(model_name, self.api_key, openai_endpoint)
        self.embedding_provider = get_embedding_provider(self.embedding_model, self.embedding_api_key, openai_endpoint)
        
        self.docs = self._load_documents()
        self.chunks = self._break_into_chunks()
        self.vector_store = self._create_vector_store()
    
    def _load_documents(self) -> List:
        """Load and validate the PDF document."""
        try:
            print("Loading PDF document...")
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from PDF")
            return docs
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def _break_into_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """Split document into chunks."""
        try:
            print("Breaking document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = text_splitter.split_documents(self.docs)
            print(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting document: {str(e)}")
    
    def _get_vector_embeddings(self, text: str) -> List[float]:
        """Get vector embeddings for text."""
        try:
            return self.embedding_provider.embed(text)
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}")
    
    def _create_vector_store(self):
        """Create FAISS vector store from document chunks."""
        try:
            print("Creating embeddings and vector store...")
            # Get embeddings for all chunks with progress bar
            embeddings = []
            for chunk in tqdm(self.chunks, desc="Creating embeddings", unit="chunk"):
                embedding = self._get_vector_embeddings(chunk.page_content)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings).astype('float32')
            
            # Create and train FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            print(f"Vector store created with {len(embeddings)} embeddings")
            return index
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def _get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """Get most relevant chunks for a query."""
        try:
            # Get query embedding
            query_embedding = self._get_vector_embeddings(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search for similar chunks
            distances, indices = self.vector_store.search(query_embedding, k)
            
            # Return relevant chunks
            return [self.chunks[i].page_content for i in indices[0]]
        except Exception as e:
            raise Exception(f"Error getting relevant chunks: {str(e)}")
    
    def ask_question(self, question: str) -> str:
        try:
            relevant_chunks = self._get_relevant_chunks(question)
            context = "\n\n".join(relevant_chunks)
            prompt = f"""Based on the following context, please answer the question. 
            If the answer cannot be found in the context, say "I cannot find the answer in the document."

            Context:
            {context}

            Question: {question}
            """
            return self.llm_provider.generate(prompt)
        except Exception as e:
            raise Exception(f"Error getting answer: {str(e)}")