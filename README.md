# 📄 Document Summarizer & Q&A Bot

Upload a PDF, generate an AI summary, and ask questions about its content using **Streamlit**, **Groq (LLaMA 3.3 70B)**, **LangChain**, and **FAISS**.

## ✨ Features

- PDF upload and processing
- AI-generated document summary
- Question answering using RAG (Retrieval-Augmented Generation)
- FAISS vector store for semantic search
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 
## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web UI |
| Groq | LLM inference (LLaMA 3.3 70B) |
| LangChain | Document loading & text splitting |
| FAISS | Vector similarity search |
| HuggingFace | Text embeddings |

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

## 👤 Author

**OptimusAutomate** — Built with ❤️ using Streamlit + Groq + LangChain
