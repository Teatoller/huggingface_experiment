# huggingfaceapi.py
"""
HuggingFace via API
No PyTorch. No dependencies issues.
Just clean API calls.
"""

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

 # Load environment variables from .env file
load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    token = os.getenv("HF_TOKEN") # Free at huggingface.co # Free at huggingface.co
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def header(title):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def divider():
    print(f"{'─' * 60}")

# Test 1: Text Generation
print("=== Text Generation ===")
result = client.chat_completion(
    messages=[{"role": "user", "content": "Explain RAG in simple terms:"}],
    model="HuggingFaceTB/SmolLM3-3B",
    max_tokens=200
)
print(result)

# Test 2: Sentiment Analysis
print("\n=== Sentiment Analysis ===")
texts = [
    "I love this bootcamp preparation!",
    "Dependency hell is frustrating.",
    "Learning AI is challenging but rewarding."
]
for text in texts:
    divider()
    result   = client.text_classification(
        text,
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    top      = result[0]           # highest-confidence label
    score    = top.score

    print(f"  Text      : {text}")
    print(f"  Sentiment : {score:.1%}")

divider()

# ── Test 3: Question Answering ─────────────────────────────────────────────────

header("Question Answering")

# Test 3: Question Answering
print("=== Question Answering ===")
context = """
LangChain is a framework for building LLM applications.
It supports multiple LLM providers including OpenAI, Anthropic,
and HuggingFace. LangChain enables building of agents, RAG systems,
and conversational AI applications.
"""
questions = [
    "What is LangChain?",
    "Which LLM providers does LangChain support?",
    "What can you build with LangChain?"
]
for question in questions:
    divider()
    result  = client.question_answering(
        question=question,
        context=context,
        model="deepset/roberta-base-squad2"
    )
    answer  = " ".join(result["answer"].split())   # collapse stray newlines
    score   = result["score"]

    print(f"  Q : {question}")
    print(f"  A : {answer}")
    print(f"      confidence: {score:.1%}")

divider()
print()