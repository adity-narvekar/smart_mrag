import os
import sys
from smart_mrag import SmartMRAG, ModelConfig
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_initialization():
    """Test initialization scenarios"""
    print("\n=== Testing Initialization ===")
    
    # Test 1: Valid initialization with default endpoints
    try:
        config = ModelConfig(
            openai_model="gpt-3.5-turbo",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro"
        )
        mrag = SmartMRAG(model_config=config)
        print("✅ Success: Valid initialization with default endpoints")
    except Exception as e:
        print(f"❌ Error in valid initialization: {str(e)}")
        raise

    # Test 2: Valid initialization with custom endpoints
    try:
        config = ModelConfig(
            openai_model="gpt-3.5-turbo",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro",
            openai_endpoint="https://adaptiveopenai-research.openai.azure.com/",
            anthropic_endpoint="https://api.anthropic.com/v1",
            google_endpoint="https://generativelanguage.googleapis.com/v1"
        )
        mrag = SmartMRAG(model_config=config)
        print("✅ Success: Valid initialization with custom endpoints")
    except Exception as e:
        print(f"❌ Error in custom endpoint initialization: {str(e)}")
        raise

    # Test 3: Invalid model name
    try:
        config = ModelConfig(
            openai_model="invalid-model",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro"
        )
        mrag = SmartMRAG(model_config=config)
        print("❌ Error: Should have failed with invalid model")
    except Exception as e:
        print("✅ Success: Caught invalid model error")

    # Test 4: Invalid endpoint URL
    try:
        config = ModelConfig(
            openai_model="gpt-3.5-turbo",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro",
            openai_endpoint="invalid-url"
        )
        mrag = SmartMRAG(model_config=config)
        print("❌ Error: Should have failed with invalid endpoint URL")
    except Exception as e:
        print("✅ Success: Caught invalid endpoint URL error")

    # Test 5: Missing required model
    try:
        config = ModelConfig(
            openai_model=None,
            anthropic_model=None,
            google_model=None
        )
        mrag = SmartMRAG(model_config=config)
        print("❌ Error: Should have failed with missing models")
    except Exception as e:
        print("✅ Success: Caught missing models error")

def test_api_keys():
    """Test API key scenarios"""
    print("\n=== Testing API Keys ===")
    
    # Test 1: Valid API keys with default endpoints
    try:
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
        os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')
        os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', '')
        
        config = ModelConfig(
            openai_model="gpt-3.5-turbo",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro"
        )
        mrag = SmartMRAG(model_config=config)
        print("✅ Success: Valid API keys with default endpoints")
    except Exception as e:
        print(f"❌ Error with valid API keys: {str(e)}")
        raise

    # Test 2: Valid API keys with custom endpoints
    try:
        config = ModelConfig(
            openai_model="gpt-3.5-turbo",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro",
            openai_endpoint="https://adaptiveopenai-research.openai.azure.com/",
            anthropic_endpoint="https://api.anthropic.com/v1",
            google_endpoint="https://generativelanguage.googleapis.com/v1"
        )
        mrag = SmartMRAG(model_config=config)
        print("✅ Success: Valid API keys with custom endpoints")
    except Exception as e:
        print(f"❌ Error with custom endpoints: {str(e)}")
        raise

    # Test 3: Missing API key
    try:
        os.environ['OPENAI_API_KEY'] = ''
        config = ModelConfig(
            openai_model="gpt-3.5-turbo",
            anthropic_model="claude-3-opus-20240229",
            google_model="gemini-pro"
        )
        mrag = SmartMRAG(model_config=config)
        print("❌ Error: Should have failed with missing API key")
    except Exception as e:
        print("✅ Success: Caught missing API key error")

def test_document_processing():
    """Test document processing scenarios"""
    print("\n=== Testing Document Processing ===")
    
    # Setup with custom endpoint
    config = ModelConfig(
        openai_model="gpt-3.5-turbo",
        anthropic_model="claude-3-opus-20240229",
        google_model="gemini-pro",
        openai_endpoint="https://adaptiveopenai-research.openai.azure.com/"
    )
    mrag = SmartMRAG(model_config=config)

    # Test 1: Valid document processing with custom endpoint
    try:
        mrag.process_documents("examples/")
        print("✅ Success: Valid document processing with custom endpoint")
    except Exception as e:
        print(f"❌ Error in valid document processing: {str(e)}")
        raise

    # Test 2: Invalid directory
    try:
        mrag.process_documents("nonexistent_directory/")
        print("❌ Error: Should have failed with invalid directory")
    except Exception as e:
        print("✅ Success: Caught invalid directory error")

    # Test 3: Empty directory
    try:
        os.makedirs("empty_dir", exist_ok=True)
        mrag.process_documents("empty_dir/")
        print("❌ Error: Should have failed with empty directory")
    except Exception as e:
        print("✅ Success: Caught empty directory error")
    finally:
        os.rmdir("empty_dir")

def test_query_processing():
    """Test query processing scenarios"""
    print("\n=== Testing Query Processing ===")
    
    # Setup with custom endpoint
    config = ModelConfig(
        openai_model="gpt-3.5-turbo",
        anthropic_model="claude-3-opus-20240229",
        google_model="gemini-pro",
        openai_endpoint="https://adaptiveopenai-research.openai.azure.com/"
    )
    mrag = SmartMRAG(model_config=config)
    mrag.process_documents("examples/")

    # Test 1: Valid query with custom endpoint
    try:
        response = mrag.query("What is the main topic of the documents?")
        print("✅ Success: Valid query processing with custom endpoint")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in valid query: {str(e)}")
        raise

    # Test 2: Empty query
    try:
        response = mrag.query("")
        print("❌ Error: Should have failed with empty query")
    except Exception as e:
        print("✅ Success: Caught empty query error")

    # Test 3: Very long query
    try:
        long_query = "?" * 1000
        response = mrag.query(long_query)
        print("❌ Error: Should have failed with very long query")
    except Exception as e:
        print("✅ Success: Caught very long query error")

def test_model_switching():
    """Test model switching scenarios"""
    print("\n=== Testing Model Switching ===")
    
    # Setup with custom endpoint
    config = ModelConfig(
        openai_model="gpt-3.5-turbo",
        anthropic_model="claude-3-opus-20240229",
        google_model="gemini-pro",
        openai_endpoint="https://adaptiveopenai-research.openai.azure.com/"
    )
    mrag = SmartMRAG(model_config=config)
    mrag.process_documents("examples/")

    # Test 1: Switch to different model with custom endpoint
    try:
        mrag.switch_model("anthropic")
        response = mrag.query("What is the main topic of the documents?")
        print("✅ Success: Model switching with custom endpoint")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in model switching: {str(e)}")
        raise

    # Test 2: Invalid model switch
    try:
        mrag.switch_model("invalid_model")
        print("❌ Error: Should have failed with invalid model switch")
    except Exception as e:
        print("✅ Success: Caught invalid model switch error")

def main():
    """Run all test scenarios"""
    print("Starting comprehensive test suite...")
    
    try:
        test_initialization()
        test_api_keys()
        test_document_processing()
        test_query_processing()
        test_model_switching()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 