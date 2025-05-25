from smart_mrag import SmartMRAG
import os

def test_basic_usage():
    # Get API key from environment or prompt user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
    
    try:
        # Initialize the reader with the sample PDF
        print("Initializing SmartMRAG...")
        reader = SmartMRAG(
            file_path="/Users/rajmohanbajaj/Projects/Mrag/smart_mrag/Microsoft.pdf",
            api_key=api_key,
            model_name="gpt-4o"  # Using GPT-4o model
        )
        
        # Print some debug information
        print(f"\nNumber of documents loaded: {len(reader.docs)}")
        print(f"Number of chunks created: {len(reader.chunks)}")
        print(f"First chunk preview: {reader.chunks[0].page_content[:200]}...")
        
        # Test questions
        questions = [
            "What was the total revenue for fiscal 2024??",
            "What is the reported net income?",
            "What were the key risks mentioned in the report?"
        ]
        
        # Ask each question
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = reader.ask_question(question)
            print(f"Answer: {answer}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_basic_usage() 