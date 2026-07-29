# 🩺 MediMatch

An AI-powered medical triage assistant for New Zealand.

## Features

- Intelligent symptom triage
- RAG-based OTC medicine recommendation
- Google Maps clinic search
- LangGraph workflow
- Streamlit UI

## Tech Stack

- Python
- LangGraph
- LangChain
- Groq LLM
- FAISS
- HuggingFace Embeddings
- Streamlit

## Project Structure

```
app/
data/
scripts/
docs/
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Create

```
.env
```

```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

### 3. Build Vector Database

```bash
uv run python scripts/build_vector_db.py
```

### 4. Run

```bash
uv run streamlit run main.py
```

## Roadmap

- [ ] Conversation Memory
- [ ] Better Retrieval
- [ ] MCP Integration
- [ ] Docker Deployment