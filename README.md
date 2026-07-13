# FinanceIQ — Bank Statement Analyzer (Hybrid RAG)

AI-powered bank statement analysis that combines **hybrid retrieval**, **deterministic Python calculations**, and **citation-enforced answers** — built so financial numbers are computed by code, never guessed by an LLM.

---

## 📊 Business Problem Statement

Reading a bank statement PDF and answering questions like *"What's my total debit this month?"* sounds simple, but it breaks most naive RAG/PDF-chat tools for a few concrete reasons:

- **LLMs are bad at arithmetic.** Ask a language model to sum 40 transaction rows from memory and it will confidently produce a wrong number — there is no way for a user to catch this without re-checking every row manually.
- **PDF tables lose their structure once converted to plain text.** Standard PDF extraction flattens a table into unstructured text, so a transaction row like `Date | Description | Debit | Credit | Balance` turns into a jumbled string, and a chunking step can literally cut a single transaction row in half across two chunks.
- **Keyword-only or embedding-only search each miss things the other would catch.** A pure vector search misses exact numeric/keyword matches (e.g. a specific transaction description); pure keyword search misses semantically related questions.
- **Answers with no source are not trustworthy for financial data.** A user (or an auditor) needs to know *which page and which chunk* a claim came from, not just a fluent-sounding answer.

### Core Objective

Build a pipeline where every number in the final answer is either (a) pulled directly from a retrieved, cited source chunk, or (b) computed deterministically by Python/pandas from the statement's raw transaction data — never generated freehand by the LLM.

---

## 🏗️ Architecture

```
User Upload PDF
      │
      ▼
┌─────────────────────┐
│  PDF Extraction      │  pdfplumber (text + tables, per page)
│  Transaction Chunking│  15 rows/chunk, a row is never split across chunks
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐    ┌──────────────┐
│ NVIDIA Embeddings    │───▶│   Pinecone   │  Vector DB
│ NemoRetriever-300m   │    └──────────────┘
└─────────────────────┘            │
                                    ▼
┌──────────────────────────────────────────┐
│         HYBRID RETRIEVAL                  │
│  BM25 (keyword) + Vector (semantic)      │
│  Combined via Reciprocal Rank Fusion (RRF)│
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│    SBERT Cross-Encoder Reranking          │
│    ms-marco-MiniLM-L-6-v2 (local)        │
│    Scores (query, doc) pairs together     │
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
    + Calculated Results (if applicable)
    + [SOURCE N] Citations
                 │
                 ▼
        RAGAS EVALUATION
   (faithfulness + answer relevancy,
    checked against a 0.7 quality gate)
```

---

## 🧠 What Makes This RAG Different: Table-Aware, Not Table-Blind

Most RAG pipelines treat a PDF as one long stream of text — a transaction table gets flattened into text, split arbitrarily by character/token count, and the original row structure is gone forever. This project is built specifically **not** to do that:

- **Row-safe chunking, not character-safe chunking.** `chunk_by_transactions()` splits transaction tables by *row count* (15 rows per chunk), so a single transaction (date, description, debit, credit, balance) is always kept fully intact inside one chunk. Free-text sections (headers, summaries) are chunked separately by line count so they don't get mixed with table rows.
- **The table isn't just embedded as flat text — the structured data is preserved alongside it.** Every table chunk stores the same rows in **two forms** in its metadata: a human-readable string (for embedding/display) and the exact `csv_data` (for computation). The tabular structure and column names are never discarded after chunking.
- **Retrieval finds the chunk, but math never comes from the LLM.** When a question needs a number, `run_calculator()` reconstructs a real pandas DataFrame directly from the `csv_data` stored in the retrieved chunks' metadata (via `pd.read_csv`), converts currency-formatted strings back into numbers, and computes totals/averages/max/min with pandas — the LLM is only ever handed the *already-correct* numbers to phrase into an answer, not asked to add them up itself.
- **Column roles are auto-detected per statement**, not hardcoded — debit/credit/balance columns are identified by matching column-name keywords (`debit`, `withdrawal`, `dr`, `spent`, `credit`, `deposit`, `cr`, `received`, `balance`), so the same calculator works across statements with different column naming conventions.

This means a question like *"What's the total debit amount?"* is answered with a number computed from the actual table data that was in the PDF — not a number an LLM inferred by reading a paragraph of flattened table text.

---

## 🏦 Why This Approach Matters for Banks, Financial Institutions & Insurance Companies

The core design choices in this project — table-safe chunking, deterministic math, and mandatory citations — map directly onto problems these industries actually have when trying to use AI on financial documents:

- **Banks & NBFCs (statement analysis, customer support, reconciliation):** Customer-facing or back-office tools that answer questions about statements (totals, balances, transaction history) cannot afford an LLM to "guess" a number. Because every calculated figure here comes from pandas running on the actual extracted transaction rows — not from the LLM doing mental math — the output is safe to use for things like balance summaries or spend breakdowns, where a wrong number has real financial consequences.
- **Financial institutions (audit & compliance):** The `[SOURCE N]` citation enforcement means every answer can be traced back to the exact page and chunk it came from. This is the kind of traceability an internal audit or compliance review needs — an answer isn't accepted just because it "sounds right," it has to point to where in the document it came from.
- **Insurance companies (claims & policy documents):** Insurance documents — claim forms, premium schedules, payout tables — are structurally similar to bank statements: rows of tabular data mixed with free text (terms, conditions, clauses). The same row-safe chunking approach (never splitting a row across chunks, preserving `csv_data` in metadata) would apply just as well to a claims table or a premium/payout schedule, so numeric fields (premium amounts, claim amounts, payout dates) stay computable rather than being flattened into unreliable text.
- **Common thread across all three:** In every one of these settings, the cost of a hallucinated number is much higher than in a general chatbot. This project's separation of "retrieval + citation" (for facts) from "pandas computation" (for numbers) is the piece that makes it realistic to trust in a regulated, numbers-heavy domain — rather than relying on an LLM's arithmetic, which is not reliable enough for financial reporting.

*Note: this project currently processes bank statement PDFs specifically (via `pdfplumber` table/text extraction). Extending it to other document types like insurance claims or policy documents would reuse the same chunking/calculator architecture, but is not something the current codebase implements out of the box.*

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Hybrid Search** | BM25 keyword search + Pinecone vector search, fused with Reciprocal Rank Fusion (RRF) |
| **SBERT Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores (query, chunk) pairs together before the final chunks are selected |
| **Deterministic Math** | Python/pandas calculates totals, averages, min/max, and opening/closing balance — no LLM guessing |
| **Citation Enforcement** | The prompt forces the LLM to tag every factual claim with `[SOURCE N]`, referencing the exact retrieved chunk |
| **Transaction-Safe Chunking** | A transaction row is never split across two chunks |
| **RAGAS Quality Evaluation** | Every answer is scored on faithfulness and answer relevancy, checked against a 0.7 threshold quality gate |
| **Streamlit Chat UI** | Upload a PDF, process it, and chat with the statement; expandable panels show retrieval details and quality scores |

---

## 📋 Prerequisites

- **Python 3.10+**
- **Pinecone** account with an index
- **NVIDIA NIM API** keys (used for both embeddings and the LLM)

---

## 🔧 Pinecone Index Setup

1. Go to the [Pinecone Console](https://app.pinecone.io/)
2. Create a new index matching the embedding model's output dimension (the embedding model used is `nvidia/llama-3.2-nemoretriever-300m-embed-v1`) with **cosine** metric
3. Copy your API key and index name into your `.env` file

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root with:

```
NVIDIA_EMBEDDING_API_KEY=nvapi-...
NVIDIA_LLM_API_KEY=nvapi-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=your-index-name
```

### 3. Run the app

```bash
streamlit run app.py
```

Then, from the sidebar: upload a bank statement PDF → click **Process PDF** → ask questions in the chat panel.

> `main.py` is a standalone CLI script covering the same extraction → chunk → embed → retrieve → answer pipeline, runnable directly from the terminal for testing outside Streamlit. It currently expects a PDF path to be set in-code (`PDF_PATH`) before running.

---

## 🗂️ Project Structure

```
Financial_rag_1/
├── .env                # API keys (not committed — see .gitignore)
├── .gitignore
├── README.md
├── requirements.txt
├── app.py              # Streamlit chat application (main entry point)
└── main.py             # CLI script covering the same pipeline
```

---

## 🏃 User Journey (Streamlit App)

1. **Upload** — Drop a bank statement PDF in the sidebar
2. **Process** — Click "Process PDF" to extract text/tables, chunk transactions, embed with NVIDIA, index into Pinecone, and build the BM25 index
3. **Ask** — Type a question, or click one of the example questions in the sidebar
4. **View** — See the answer with inline `[SOURCE N]` citations, and calculated results (if the question needed math)
5. **Expand** — Open "Retrieval Pipeline Details" to see which chunks were retrieved and their rerank scores
6. **Quality Check** — Open "RAGAS Quality Scores" to see faithfulness and answer-relevancy scores for that answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **App / UI** | Streamlit |
| **PDF Parsing** | pdfplumber |
| **Embeddings** | NVIDIA `nemoretriever-300m` (via `langchain-nvidia-ai-endpoints`) |
| **LLM** | Qwen3.5-122B-A10B via NVIDIA NIM |
| **Vector DB** | Pinecone |
| **Keyword Search** | BM25Okapi (`rank-bm25`) |
| **Reranker** | SBERT `ms-marco-MiniLM-L-6-v2` (`sentence-transformers` CrossEncoder) |
| **Calculator** | pandas |
| **Evaluation** | RAGAS (faithfulness, answer relevancy) |
| **Orchestration** | LangChain (prompts, output parsing) |

---

## 📌 What I Learned

- Designing chunking logic around the *shape of the data* (transaction rows) instead of a fixed character/token window, so structured records survive the retrieval pipeline intact.
- Building a hybrid retriever from scratch (BM25 + vector search) and fusing results with Reciprocal Rank Fusion instead of relying on a single retrieval method.
- Keeping deterministic computation (pandas) and generative text (LLM) strictly separated so the two failure modes — wrong retrieval vs. wrong arithmetic — can be debugged independently.
- Enforcing inline citations at the prompt level so every claim in an answer is traceable back to a specific chunk.
- Using RAGAS to score answers automatically instead of only trusting that an answer "looks right."

---

## 📄 License

MIT
