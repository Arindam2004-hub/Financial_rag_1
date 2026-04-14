# FinanceIQ — Bank Statement Analyzer

> AI-powered financial document analysis with **Hybrid RAG**, **deterministic calculations**, and **citation-enforced answers**.

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🏗️ Architecture

```
User Upload PDF
      │
      ▼
┌─────────────────────┐
│  PDF Extraction      │  pdfplumber (text + tables)
│  Transaction Chunking│  15 rows/chunk, never splits a row
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐    ┌──────────────┐
│ NVIDIA Embeddings    │───▶│   Pinecone   │  Vector DB (1024-dim)
│ NemoRetriever-300m   │    └──────────────┘
└─────────────────────┘            │
                                    ▼
┌──────────────────────────────────────────┐
│         HYBRID RETRIEVAL                  │
│  BM25 (keyword) + Vector (semantic)      │
│  Combined via Reciprocal Rank Fusion     │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│    SBERT Cross-Encoder Reranking          │
│    ms-marco-MiniLM-L-6-v2 (local)       │
│    Scores (query, doc) pairs together    │
└────────────────┬─────────────────────────┘
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
┌───────────────┐ ┌───────────────┐
│ Python        │ │ Qwen3.5-122B  │
│ Calculator    │ │ LLM w/        │
│ (pandas math) │ │ [SOURCE N]    │
│ Deterministic │ │ Citations     │
└───────────────┘ └───────────────┘
          │             │
          └──────┬──────┘
                 ▼
          FINAL ANSWER
    + Calculated Results
    + Source Citations
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | BM25 keyword + Vector semantic search with RRF fusion |
| **SBERT Reranking** | Cross-encoder scores (query, doc) pairs for precision |
| **Deterministic Math** | Python/pandas calculates totals, averages, balances — no LLM guessing |
| **Citation Enforcement** | LLM forced to cite `[SOURCE N]` for every factual claim |
| **Transaction-Safe Chunks** | Never splits a transaction row across two chunks |
| **Dark Theme Dashboard** | Professional financial UI with glassmorphism and micro-animations |

---

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- **Pinecone** account with an index (1024 dimensions, cosine metric)
- **NVIDIA NIM API** keys (for embeddings and LLM)

---

## 🔧 Pinecone Index Setup

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new index:
   - **Name:** `financial-rag`
   - **Dimensions:** `1024`
   - **Metric:** `cosine`
   - **Cloud:** Any (AWS/GCP/Azure)
3. Copy your API key

---

## 🚀 Quick Start

### 1. Clone & Configure Environment

```bash
cd new_const_project

# Create .env file in project root with your keys:
# NVIDIA_EMBEDDING_API_KEY=nvapi-...
# NVIDIA_LLM_API_KEY=nvapi-...
# PINECONE_API_KEY=pcsk_...
# PINECONE_INDEX_NAME=financial-rag
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies API calls to backend)
npm run dev
```

The app will be available at `http://localhost:5173`

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with service status |
| `/api/upload` | POST | Upload PDF → extract → chunk → embed → index |
| `/api/chat` | POST | `{ "question": "...", "session_id": "..." }` |
| `/api/history` | GET | Get chat history for a session |
| `/api/history` | DELETE | Clear chat history |

### Example Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total debit amount?", "session_id": "default"}'
```

### Response Structure

```json
{
  "answer": "The total debit amount is ₹1,25,000.00 [SOURCE 1] ...",
  "calculated": "=== CALCULATED RESULTS (Python Math) ===\nTotal Debits: 125000.00\n...",
  "retrieval": {
    "hybrid_candidates": 15,
    "reranked_count": 5,
    "sources": [
      {
        "chunk_index": 1,
        "page": 2,
        "chunk_type": "transaction_rows",
        "rerank_score": 4.2341,
        "content_preview": "Date | Description | Debit..."
      }
    ]
  },
  "needs_calculation": true
}
```

---

## 🗂️ Project Structure

```
new_const_project/
├── .env                        # API keys (NVIDIA, Pinecone)
├── README.md
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app with endpoints
│   ├── config.py               # Environment configuration
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── pdf_service.py      # PDF extraction & chunking
│       ├── rag_service.py      # Hybrid retrieval + reranking
│       ├── calculator.py       # Deterministic Python calculations
│       └── llm_service.py      # LLM prompting with citations
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx             # Root app component
│       ├── index.css           # Design system (dark theme)
│       ├── api/
│       │   └── client.js       # Axios API client
│       └── components/
│           ├── Sidebar.jsx     # Upload + pipeline status
│           ├── ChatPanel.jsx   # Chat interface
│           ├── ChatMessage.jsx # Message with citations
│           ├── FileUpload.jsx  # Drag-and-drop upload
│           └── LoadingStates.jsx # Thinking indicators
├── main.py                     # Original CLI prototype
└── app.py                      # Original Streamlit prototype
```

---

## 🏃 Full User Journey

1. **Upload** — Drop a bank statement PDF in the sidebar
2. **Process** — Click "Process & Index" to extract, chunk, embed, and index
3. **Ask** — Type questions like "What is the total debit amount?"
4. **View** — See the answer with inline `[SOURCE N]` citations
5. **Expand** — Click source citations to see which chunks were retrieved
6. **Calculate** — Any math question triggers deterministic Python calculations

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Vite, Vanilla CSS |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Embeddings** | NVIDIA NemoRetriever-300m (1024-dim) |
| **LLM** | Qwen3.5-122B-A10B via NVIDIA NIM |
| **Vector DB** | Pinecone (serverless) |
| **Keyword Search** | BM25Okapi (rank-bm25) |
| **Reranker** | SBERT ms-marco-MiniLM-L-6-v2 |
| **PDF Parser** | pdfplumber |
| **Calculator** | Python pandas |

---

## 📄 License

MIT
