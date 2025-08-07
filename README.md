# 🔍 Tokenizer Cost Analyzer — Compare LLM Tokenization & ₹ Cost!

Ever wondered how much your prompt _really_ costs when sent to GPT-4, LLaMA, Claude, or Mistral?

Welcome to **Tokenizer Cost Analyzer** — a sleek Streamlit app that breaks down your prompt into tokens, compares how different models tokenize the same text, and shows you the estimated ₹ cost per model 💸.

---

## ⚙️ What This App Does

- 🧩 **Tokenizes input text** using different model tokenizers (OpenAI GPT-4, LLaMA2, Mistral, Claude).
- 📊 **Estimates token count and cost** based on real-world pricing (₹ per 1K tokens).
- 🧠 **Visualizes** which model is more efficient using token count and cost bar charts.
- 💡 Helps you **optimize your prompts** by choosing models that tokenize more efficiently.

---

## 🕹️ How to Use

1. **Enter your sentence** in the text box (e.g., _"I am becoming an AI thought leader."_).
2. Click **“Analyze Tokenization”**.
3. Instantly see:
   - Model-wise token breakdown 🧩
   - Token IDs 🔢
   - Estimated cost in ₹ 💰
   - Clean side-by-side comparison table
   - Beautiful charts comparing token count & cost

---

## 📸 Preview

| Model | Tokens              | Token IDs        | Token Count | ₹ Cost    |
| ----- | ------------------- | ---------------- | ----------- | --------- |
| GPT-4 | `['I', ' am', ...]` | `[40, 539, ...]` | `9`         | `₹0.0054` |

![Sample UI](https://your-screenshot-url.com/sample.png) <!-- Replace with actual screenshot URL if available -->

---

## 🚀 Run It Locally

# Clone the repo

git clone https://github.com/your-username/tokenizer-cost-analyzer.git](https://github.com/AIExecution/TokenizerCostAnalyzer.git
cd tokenizer-cost-analyzer

# Install dependencies

pip install -r requirements.txt

# Add Hufgging Face token value

Make sure you agree to terms and conditions for below models in hugging face otherwise the token might not be useful.
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

Replace hf_your_actual_token_here with your actual token in .env file.

HF_TOKEN=hf_your_actual_token_here

# Run the app

streamlit run app.py

# If you want to check Tokeniser for specific model and get their computation details run

streamlit run tokenizer_decoder_app.py

## 🌐 Run It on Hugging Face Spaces (No Setup Needed)

Just click **“Open in Spaces”** (if deployed), or **fork and deploy your own**:

---

## 🧠 Powered By

- 🤖 [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- 🧮 [tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- 🎨 [matplotlib](https://matplotlib.org/)
- 📊 [plotly](https://plotly.com/python/)
- 🧱 [Streamlit](https://streamlit.io)

---

## 📬 Contact & Credits

Built with ❤️ by **Ashwin Shah** — feel free to fork, remix, or contribute!

---

## 📜 License

**MIT License** — free to use and adapt!
