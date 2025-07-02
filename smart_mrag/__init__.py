import os
from typing import Optional, List
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import fitz
from PIL import Image
import base64
import io
from .utils import ModelConfig
from tqdm import tqdm

__version__ = "0.1.3"

# Only OpenAI models
RECOMMENDED_MODELS = {
    "openai": {
        "llm_models": [
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
        "requires": ["openai_api_key"]
    }
}

def get_recommended_models():
    return RECOMMENDED_MODELS

def get_required_api_keys(llm_model, embedding_model):
    required_keys = set()
    if any(model in llm_model for model in ["gpt-3.5", "gpt-4"]):
        required_keys.add("openai_api_key")
    if any(model in embedding_model for model in ["text-embedding"]):
        required_keys.add("openai_api_key")
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
        }
    }

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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set OPENAI_API_KEY environment variable")
        if embedding_model:
            if model_name in self.DEFAULT_MODELS and embedding_model != self.DEFAULT_MODELS[model_name]["embedding_model"]:
                if not embedding_api_key:
                    raise ValueError(f"Embedding API key is required when using custom embedding model: {embedding_model}")
                self.embedding_api_key = embedding_api_key
            else:
                self.embedding_api_key = self.api_key
        else:
            if model_name in self.DEFAULT_MODELS:
                embedding_model = self.DEFAULT_MODELS[model_name]["embedding_model"]
            else:
                raise ValueError("Embedding model is required when using a non-default model")
            self.embedding_api_key = self.api_key
        self.embedding_model = embedding_model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=openai_endpoint if openai_endpoint else None
        )
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model_name,
            base_url=openai_endpoint if openai_endpoint else None
        )
        self.embedding_client = OpenAI(
            api_key=self.embedding_api_key,
            base_url=openai_endpoint if openai_endpoint else None
        )
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
            response = self.embedding_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return [r.embedding for r in response.data][0]
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
        """
        Ask a question about the document.
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The answer to the question
        """
        try:
            # Get relevant chunks
            relevant_chunks = self._get_relevant_chunks(question)
            
            # Create context from chunks
            context = "\n\n".join(relevant_chunks)
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. 
            If the answer cannot be found in the context, say "I cannot find the answer in the document."

            Context:
            {context}

            Question: {question}
            """
            
            # Get answer from LLM
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            return response.content
        except Exception as e:
            raise Exception(f"Error getting answer: {str(e)}") 