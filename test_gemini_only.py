from dotenv import load_dotenv
load_dotenv()

import os
from smart_mrag import SmartMRAG

def main():
    google_key = os.getenv("GOOGLE_API_KEY")

    file_path = "AAPL10K.pdf"

    reader = SmartMRAG(
        file_path=file_path,
        api_key=google_key,
        model_name="gemini-3.6-flash",
        embedding_model="gemini-embedding-001"
    )

    question = "What was the total revenue?"
    answer = reader.ask_question(question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()