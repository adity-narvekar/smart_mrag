# Smart MRAG

A smart Multi-modal RAG (Retrieval Augmented Generation) library for processing PDFs with various language models.

## Installation

```bash
pip install smart_mrag
```

## Usage

```python
from smart_mrag import SmartMRAG

# Initialize with default OpenAI models
reader = SmartMRAG(
    file_path="path/to/your/file.pdf",
    api_key="your-api-key",  # Optional if set in environment variables
    model_name="gpt-4",  # Optional, defaults to "gpt-4"
)

# Or initialize with custom models
reader = SmartMRAG(
    file_path="path/to/your/file.pdf",
    api_key="your-api-key",
    model_name="your-custom-model",
    embedding_model="your-embedding-model",  # Required for custom models
    embedding_api_key="your-embedding-api-key"  # Optional, only needed if different from main API key
)

# Ask questions about the document
while True:
    question = input("Enter your question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    try:
        answer = reader.ask_question(question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")
```

## Features

- Process PDF documents with text and images
- Support for both default OpenAI models and custom models
- Vector embeddings for efficient document retrieval
- Error handling for common issues
- Automatic model and embedding model compatibility checking

## Default Models

The library provides default support for the following OpenAI model combinations:

- gpt-4 (default) with text-embedding-ada-002
- gpt-4-turbo with text-embedding-ada-002
- gpt-3.5-turbo with text-embedding-ada-002

For these default models, the embedding model is automatically set to text-embedding-ada-002 unless specified otherwise.

## Custom Models

You can use any custom model by providing:
- The model name
- A compatible embedding model
- Appropriate API keys

When using custom models, you must provide:
1. The model name
2. A compatible embedding model
3. The appropriate API key(s)

## Requirements

- Python 3.8 or higher
- API key(s) for the model(s) you want to use
- Required Python packages (automatically installed with pip)

## Error Handling

The library handles various errors including:
- Invalid file paths
- Invalid API keys
- Missing embedding models for custom models
- Network errors
- PDF processing errors
- Model and embedding model compatibility issues
- Missing embedding API keys for custom models 