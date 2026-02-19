# huggingface_trial.py
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

 # Load environment variables from .env file
load_dotenv()


# Free API calls to HuggingFace models
client = InferenceClient(
    provider="hf-inference",
    token = os.getenv("HF_TOKEN") # Free at huggingface.co
)

# Text generation — gpt2 is small and reliably warm on free tier
# (hf-inference no longer hosts gpt2/Mistral for text_generation as of 2025)
result = client.chat_completion(
    messages=[{"role": "user", "content": "The future of AI is"}],
    model="HuggingFaceTB/SmolLM3-3B",
    max_tokens=100
)
print(result)

# Text classification
result = client.text_classification(
    "I love building AI applications!",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
print(result)

# Question answering
result = client.question_answering(
    question="What is RAG?",
    context="RAG is Retrieval Augmented Generation, a technique that combines "
            "retrieval of relevant documents with language model generation.",
    model="deepset/roberta-base-squad2"
)
print(result)