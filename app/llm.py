import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=os.getenv("QWEN_API_KEY")
)

MODEL = os.getenv("MODEL_NAME")

def ask_llm(context, question):
    prompt = f"""
You are a helpful assistant.

Use the context to answer the question.

IMPORTANT:
- Always include sources at the end.
- Sources must include document name and page number.

Context:
{context}

Question:
{question}

Answer format:
- Answer
- Sources:
    - document_name (page X)
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content