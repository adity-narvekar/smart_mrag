from smart_mrag import SmartMRAG
import os

def main():
    # Get API key from environment or prompt user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your API key: ")
    
    # Get model choice from user
    print("\nAvailable default models:")
    print("1. gpt-4 (default)")
    print("2. gpt-4-turbo")
    print("3. gpt-3.5-turbo")
    print("4. Custom model")
    model_choice = input("\nEnter model number (1-4) or press Enter for default: ")
    
    model_map = {
        "1": "gpt-4",
        "2": "gpt-4-turbo",
        "3": "gpt-3.5-turbo"
    }
    
    if model_choice == "4":
        model_name = input("Enter your custom model name: ")
        # For custom models, we need embedding model
        embedding_model = input("Enter embedding model name: ")
        embedding_api_key = input("Enter API key for embedding model (press Enter to use same as main API key): ")
        if not embedding_api_key:
            embedding_api_key = api_key
    else:
        model_name = model_map.get(model_choice, "gpt-4")
        # For default models, embedding model is optional
        use_custom_embedding = input("\nDo you want to use a custom embedding model? (y/n): ").lower() == 'y'
        embedding_model = None
        embedding_api_key = None
        
        if use_custom_embedding:
            embedding_model = input("Enter custom embedding model name: ")
            embedding_api_key = input("Enter API key for embedding model (press Enter to use same as main API key): ")
            if not embedding_api_key:
                embedding_api_key = api_key
    
    # Initialize the reader
    try:
        reader = SmartMRAG(
            file_path="path/to/your/file.pdf",
            api_key=api_key,
            model_name=model_name,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key
        )
        
        # Interactive question answering
        print("\nEnter your questions about the document. Type 'quit' to exit.")
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'quit':
                break
            
            try:
                answer = reader.ask_question(question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    except Exception as e:
        print(f"Error initializing reader: {str(e)}")

if __name__ == "__main__":
    main() 