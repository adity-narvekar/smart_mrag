from smart_mrag import SmartMRAG
import os

def main():
    # Get API key from environment or prompt user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
    
    # Initialize the reader
    try:
        reader = SmartMRAG(
            file_path="AAPL10K.pdf",  # Using the sample PDF
            api_key=api_key
        )
        
        # Test with a simple question
        question = "What was Apple's revenue in the last fiscal year?"
        print(f"\nQuestion: {question}")
        answer = reader.ask_question(question)
        print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 