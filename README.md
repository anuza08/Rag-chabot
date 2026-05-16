# Chat with your Documents — RAG App

A minimal Retrieval-Augmented Generation (RAG) app. Upload PDFs or text files, ask questions in natural language, get answers grounded in **your** documents — with sources cited.

Built as a learning project to understand how modern AI knowledge systems work under the hood.

---

## What you'll learn from this project

- **Document chunking** — why and how long documents get split before going to an LLM
- **Embeddings** — turning text into vectors that capture meaning
- **Vector databases** — storing and searching those vectors efficiently (semantic search)
- **Retrieval** — pulling the most relevant chunks for a given question
- **Prompt construction** — stuffing retrieved context into the prompt to get grounded answers
- **Deployment** — shipping a Streamlit app to Hugging Face Spaces for free

---

## Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| UI | Streamlit | Fastest way to ship a Python web app |
| Orchestration | LangChain | Glues the RAG steps together cleanly |
| Embeddings | `BAAI/bge-small-en-v1.5` | Free, runs on CPU, strong quality |
| Vector DB | Chroma | Free, runs locally, zero setup |
| LLM | OpenAI GPT-4o-mini | Cheap and fast; swap for any provider |

---

## Run it locally

```bash
# 1. Set up the environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# edit .env and paste your OpenAI key

# 3. Run
streamlit run app.py
```

Open <http://localhost:8501>, upload a PDF, and start chatting.

---

## Deploy to Hugging Face Spaces (free)

1. Create a new Space → <https://huggingface.co/new-space>
2. Choose **Streamlit** as the SDK
3. Push these files to the Space's git repo:
   - `app.py`
   - `requirements.txt`
4. In the Space's **Settings → Variables and secrets**, add `OPENAI_API_KEY`
5. Your app is now live at `https://huggingface.co/spaces/<you>/<name>`

First build takes ~5 minutes (downloads the embedding model). Subsequent restarts are fast.

---

## How it works (in one picture)

```
INGESTION (one-time, when you upload):
  Documents ──▶ Chunks ──▶ Embeddings ──▶ Vector DB

QUERY (every question):
  Question ──▶ Embedding ──▶ Similarity search ──▶ Top-K chunks ──▶ LLM ──▶ Answer
```

---

## Ideas to extend this

Once the basics click, try one of these to go deeper:

- Add web-page ingestion (paste a URL → scrape → embed)
- Swap Chroma for **Qdrant** or **Pinecone** (production-grade vector DBs)
- Add a **re-ranker** (cross-encoder) to improve retrieval quality
- Swap OpenAI for a **local LLM** via Ollama
- Add **conversation memory** for multi-turn Q&A
- Show **inline citations** in the answer text, not just below it
- Wrap it as an **MCP server** so any AI assistant can query your docs

---

## Project structure

```
rag-doc-chat/
├── app.py              # The whole app, ~200 lines
├── requirements.txt    # Python deps
├── .env.example        # Template for API key
├── .gitignore
└── README.md
```

---

Built as a hands-on project to learn RAG, embeddings, and vector databases. PRs and forks welcome.
