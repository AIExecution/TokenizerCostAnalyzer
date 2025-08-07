import streamlit as st
from transformers import AutoTokenizer
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

# Setup
HF_TOKEN = os.getenv("HF_TOKEN")
# os.environ["HF_TOKEN"] = ""

print("-----------------", tiktoken.list_encoding_names())

# Define available models and their pricing (₹ per 1K tokens)
MODELS = {
    "OpenAI GPT-4 (tiktoken)": {"tokenizer": "tiktoken:gpt-4", "cost_per_1k": 0.60},
    "LLaMA 2 (HuggingFace)": {
        "tokenizer": "meta-llama/Llama-2-7b-chat-hf",
        "cost_per_1k": 0.20,
    },
    "Mistral 7B (HuggingFace)": {
        "tokenizer": "mistralai/Mistral-7B-Instruct-v0.1",
        "cost_per_1k": 0.15,
    },
    "Claude (Simulated)": {
        "tokenizer": "gpt2",  # Placeholder for Claude – public tokenizer not available
        "cost_per_1k": 0.30,
    },
}
# Title
st.title("🔍 Tokenizer Decoder: Understand Your LLM Token Costs")
# Input
token_ids = ""
token_count = 0
input_text = st.text_area(
    "✍️ Enter your sentence:", "I am becoming an AI thought leader."
)
selected_model = st.selectbox("🤖 Choose a model:", list(MODELS.keys()))
if st.button("🔍 Analyze Tokenization"):
    tokenizer_info = MODELS[selected_model]
    tokenizer_id = tokenizer_info["tokenizer"]
    cost_per_k = tokenizer_info["cost_per_1k"]
    # Tokenize
    if tokenizer_id.startswith("tiktoken"):
        encoding_name = tokenizer_id.split(":")[1]
        encoding = tiktoken.encoding_for_model("gpt-4")
        # encoding = tiktoken.get_encoding(encoding_name)
        token_ids = encoding.encode(input_text)
        tokens = [encoding.decode([tid]) for tid in token_ids]
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        tokenized = tokenizer(input_text, return_tensors="pt")
        token_ids = tokenized["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # Cost
    token_count = len(token_ids)
    estimated_cost = round((token_count / 1000) * cost_per_k, 5)
    cost_1m = round((1000000 / 1000) * cost_per_k, 2)
    # Output
    st.markdown("### 🧩 Tokenized Output")
    st.write(tokens)
    st.markdown("### 🔢 Token IDs")
    st.write(token_ids)
    st.markdown("### 📊 Summary")
    st.metric("Token Count", token_count)
    st.metric("Estimated Cost (₹)", f"{estimated_cost}")
    st.metric("Cost for 1M Tokens (₹)", f"{cost_1m}")
    st.markdown("### 🧠 Insight")
    st.info(
        f"Same sentence → different token splits across models → different ₹ cost. Optimize wisely!"
    )
