from dotenv import load_dotenv
load_dotenv()

import os
from smart_mrag import SmartMRAG

def main():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")  # needed for embeddings — Claude has no embedding API

    file_path = "AAPL10K.pdf"

    reader = SmartMRAG(
        file_path=file_path,
        api_key=anthropic_key,
        model_name="claude-sonnet-4-5",
        embedding_model="text-embedding-ada-002",
        embedding_api_key=openai_key
    )

    question = "What was the total revenue?"
    answer = reader.ask_question(question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()