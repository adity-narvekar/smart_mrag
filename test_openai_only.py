from smart_mrag import SmartMRAG
import os

def main():
    # Get OpenAI API key from environment or prompt user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")

    # Path to your PDF file
    file_path = "/Users/rajmohanbajaj/Projects/Mrag/smart_mrag/AAPL10K.pdf"  # Change this to your PDF file

    # Initialize SmartMRAG with OpenAI only
    reader = SmartMRAG(
        file_path=file_path,
        api_key=api_key,
        model_name="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo"
        embedding_model="text-embedding-ada-002"
    )

    # Ask a question
    question = "What was the total revenue ?"
    answer = reader.ask_question(question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main() 