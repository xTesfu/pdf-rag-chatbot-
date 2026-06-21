import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

api_key = os.getenv("QWEN_API_KEY")
model_name = os.getenv("MODEL_NAME")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=api_key
)

MODEL = model_name

def ask_llm(context, question):
    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content