import streamlit as st
from transformers import AutoTokenizer
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# Setup
HF_TOKEN = os.getenv("HF_TOKEN")
# os.environ["HF_TOKEN"] = ""
st.set_page_config(layout="wide")

# Model configs
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
    "Claude (Simulated)": {"tokenizer": "gpt2", "cost_per_1k": 0.30},
}

st.title("🔍 Tokenizer Decoder: Understand Your LLM Token Costs")

input_text = st.text_area(
    "✍️ Enter your sentence:", "I am becoming an AI thought leader."
)
if st.button("🔍 Analyze Tokenization"):

    results = []
    for model_name, model_info in MODELS.items():
        tokenizer_id = model_info["tokenizer"]
        cost_per_k = model_info["cost_per_1k"]

        if tokenizer_id.startswith("tiktoken"):
            encoding = tiktoken.encoding_for_model("gpt-4")
            token_ids = encoding.encode(input_text)
            tokens = [encoding.decode([tid]) for tid in token_ids]
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, use_fast=True, token=os.getenv("HF_TOKEN")
            )
            tokenized = tokenizer(input_text, return_tensors="pt")
            token_ids = tokenized["input_ids"][0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

        filtered_tokens = [t for t in tokens if t.strip() != ""]
        filtered_token_ids = [
            token_ids[i] for i, t in enumerate(tokens) if t.strip() != ""
        ]

        if filtered_tokens:
            token_count = len(filtered_tokens)
            estimated_cost = round((token_count / 1000) * cost_per_k, 5)
            results.append(
                {
                    "Model": model_name,
                    "Token Count": token_count,
                    "Tokens": filtered_tokens,
                    "Token IDs": filtered_token_ids,
                    "Estimated Cost (₹)": estimated_cost,
                }
            )

    df = pd.DataFrame(results)
    st.markdown("### 📊 Token Analysis Table")
    st.dataframe(df, use_container_width=True)

    cheapest_model = df.loc[df["Estimated Cost (₹)"].idxmin()]["Model"]
    st.success(f"💸 Most Cost-Efficient Model: **{cheapest_model}**")

    st.markdown("### 📈 Cost vs Token Count (per model)")

    # Setup side-by-side columns
    # Setup side-by-side columns
    col1, col2 = st.columns([1, 1])

    # Chart 1: Estimated Cost (Green)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(3, 2))
        bars = ax1.bar(df["Model"], df["Estimated Cost (₹)"], color="green")
        ax1.set_ylabel("₹", fontsize=10)
        ax1.set_xlabel("Model", fontsize=10)
        ax1.set_title("Estimated Cost", fontsize=12)
        ax1.set_xticks(range(0, len(df)))
        ax1.set_xticklabels(df["Model"], rotation=45, fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        st.pyplot(fig1)

    # Chart 2: Token Count (Orange)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(3, 2))
        bars = ax2.bar(df["Model"], df["Token Count"], color="orange")
        ax2.set_ylabel("Tokens", fontsize=10)
        ax2.set_xlabel("Model", fontsize=10)
        ax2.set_title("Token Count", fontsize=12)
        ax2.set_xticks(range(0, len(df)))
        ax2.set_xticklabels(df["Model"], rotation=45, fontsize=8)
        ax2.tick_params(axis="y", labelsize=8)
        st.pyplot(fig2)
    st.info("🧠 Tip: Use token-efficient models to reduce cost without losing quality.")
