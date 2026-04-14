

import os
import re
import pdfplumber
import pandas as pd
from io import StringIO
from dotenv import load_dotenv


from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from pinecone import Pinecone,ServerlessSpec


load_dotenv()


NVIDIA_EMBEDDING_API_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")
NVIDIA_LLM_API_KEY       = os.getenv("NVIDIA_LLM_API_KEY")
PINECONE_API_KEY         = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME      = os.getenv("PINECONE_INDEX_NAME")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc    = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(
    name="financial-analyst-rag",
    dimension=2048,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


# NVIDIA MODELS


def get_nvidia_embeddings() -> NVIDIAEmbeddings:

    return NVIDIAEmbeddings(
        model   = "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        api_key = NVIDIA_EMBEDDING_API_KEY,
        truncate= "NONE",

    )
    test_vector = embeddings.embed_query("test")
    print(f"✅ Actual embedding dimension: {len(test_vector)}")

    return embeddings






def get_qwen_llm() -> ChatNVIDIA:

    return ChatNVIDIA(
        model       = "qwen/qwen3.5-122b-a10b",
        api_key     = NVIDIA_LLM_API_KEY,
        temperature = 0.0,
        max_tokens  = 2048,
    )



# PDF EXTRACTION


def extract_pdf_content(pdf_path: str) -> dict:

    text_docs  = []
    dataframes = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # Extract plain text (headers, account info, summaries)
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_docs.append(Document(
                    page_content = page_text,
                    metadata     = {
                        "source": pdf_path,
                        "page"  : page_num + 1,
                        "type"  : "text"
                    }
                ))

            # Extract tables as DataFrames
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                df = pd.DataFrame(table[1:], columns=table[0])
                # table[0]  → first row = column headers
                # table[1:] → remaining rows = actual data

                df = df.dropna(how='all')  # remove fully empty rows
                df = df.fillna('')         # replace NaN with empty string

                dataframes.append({
                    "df"    : df,
                    "page"  : page_num + 1,
                    "source": pdf_path
                })

    print(f" Extracted {len(text_docs)} text sections "
          f"and {len(dataframes)} tables")
    return {"text_docs": text_docs, "dataframes": dataframes}



 # TRANSACTION-AWARE CHUNKING


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Fix NaN column names
    new_columns = []
    for i, col in enumerate(df.columns):
        if pd.isna(col) or str(col).strip() == '' or str(col) == 'nan':
            new_columns.append(f"col_{i}")
            # Replace empty/NaN column name with "col_0", "col_1" etc.
        else:
            new_columns.append(str(col).strip())
    df.columns = new_columns

    # Fix special characters inside cells
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('\n', ' ', regex=False)   # newlines → space
            .str.replace('\r', ' ', regex=False)   # carriage returns → space
            .str.replace('\t', ' ', regex=False)   # tabs → space
            .str.replace('"', "'", regex=False)    # double quotes → single
            .str.strip()
        )
    return df


def clean_metadata_value(value) -> str:

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    # Remove characters that break Pinecone metadata JSON parsing
    text = text.replace('\x00', '')   # null bytes
    text = text.replace('\r', ' ')    # carriage returns
    # Pinecone metadata values have a size limit — truncate if too long
    return text[:10000]


def chunk_by_transactions(extracted: dict,
                          rows_per_chunk: int = 15) -> list[Document]:

    all_chunks = []

    # Text pages: chunk by lines (20 lines per chunk)
    for doc in extracted["text_docs"]:
        lines = doc.page_content.split("\n")
        for i in range(0, len(lines), 20):
            chunk_text = "\n".join(lines[i: i + 20])
            if chunk_text.strip():
                all_chunks.append(Document(
                    page_content = chunk_text,
                    metadata     = {**doc.metadata, "chunk_type": "text"}
                ))

    # Transaction tables: chunk by ROWS
    for table_info in extracted["dataframes"]:
        df     = table_info["df"]
        page   = table_info["page"]
        source = table_info["source"]

        #   Clean the DataFrame before chunking
        df = clean_dataframe(df)

        for start_row in range(0, len(df), rows_per_chunk):
            end_row  = min(start_row + rows_per_chunk, len(df))
            chunk_df = df.iloc[start_row:end_row]
            # iloc[start:end] slices exactly those rows by position
            # A transaction row is NEVER split across two chunks

            # Get readable text and CSV for this chunk
            chunk_text = chunk_df.to_string(index=False)
            chunk_csv  = chunk_df.to_csv(index=False)

            #   Clean ALL metadata values before sending to Pinecone
            # Pinecone rejects NaN, None, newlines, and oversized values
            all_chunks.append(Document(
                page_content = chunk_text,
                metadata     = {
                    "source"    : clean_metadata_value(source),
                    "page"      : int(page),       # ensure it's a plain int
                    "type"      : "table",
                    "chunk_type": "transaction_rows",
                    "start_row" : int(start_row),  # ensure plain int
                    "end_row"   : int(end_row),    # ensure plain int
                    "csv_data"  : clean_metadata_value(chunk_csv),

                    "columns"   : clean_metadata_value(
                                    ", ".join(df.columns.tolist()))
                    # Store columns as a comma-separated string (not a list)
                    # Pinecone does not support list values in metadata
                }
            ))

    print(f" Created {len(all_chunks)} transaction-safe chunks")
    return all_chunks



#  EMBED AND STORE IN PINECONE


def index_documents(documents: list[Document]) -> PineconeVectorStore:
    """
    Converts all chunks into 1024-dimensional vectors using NVIDIA embeddings
    and stores them in your Pinecone index for fast similarity search.

    Your Pinecone index MUST have dimensions=1024 to match NVIDIA model.
    """
    embeddings  = get_nvidia_embeddings()
    vectorstore = PineconeVectorStore.from_documents(
        documents  = documents,
        embedding  = embeddings,
        index_name = PINECONE_INDEX_NAME
    )
    print(f"✅ Indexed {len(documents)} chunks into Pinecone")
    return vectorstore



#  CALCULATOR


def run_calculator(retrieved_docs: list[Document]) -> str:
    """
    Reads the raw CSV data stored in retrieved chunk metadata
    and runs real Python/pandas calculations.

    WHY NOT LET THE LLM CALCULATE?
    LLMs are bad at arithmetic — they guess and get wrong totals.
    This function uses actual Python math: sum(), mean(), max(), min()
    So the numbers are always 100% accurate.

    Called automatically when the question contains calculation keywords.
    """

    # Collect all CSV data from retrieved table chunks
    all_rows = []
    for doc in retrieved_docs:
        if doc.metadata.get("chunk_type") == "transaction_rows":
            csv_data = doc.metadata.get("csv_data", "")
            if csv_data:
                try:
                    chunk_df = pd.read_csv(StringIO(csv_data))
                    all_rows.append(chunk_df)
                except Exception:
                    continue

    if not all_rows:
        return ""
    # Return empty string if no table data found
    # The LLM will then answer from text context alone

    # Combine all retrieved chunks into one DataFrame
    combined_df = pd.concat(all_rows, ignore_index=True)


    # Clean columns — remove currency symbols and convert to numbers
    for col in combined_df.columns:
        combined_df[col] = pd.to_numeric(
            combined_df[col]
                .astype(str)
                .str.replace(',', '',  regex=False)  # "1,500" → "1500"
                .str.replace('₹', '',  regex=False)  # remove ₹
                .str.replace('$', '',  regex=False)  # remove $
                .str.replace('£', '',  regex=False)  # remove £
                .str.strip(),
            errors='coerce'

        )

    # Identify debit, credit, balance columns automatically
    debit_col = credit_col = balance_col = None
    for col in combined_df.columns:
        if col is None:
            continue
        c = str(col).lower().strip()
        if any(x in c for x in ['debit', 'withdrawal', 'dr', 'spent']):
            debit_col = col
        if any(x in c for x in ['credit', 'deposit', 'cr', 'received']):
            credit_col = col
        if 'balance' in c:
            balance_col = col

    # Build results string
    lines = [f"=== CALCULATED RESULTS (Python Math) ===",
             f"Transactions Analyzed : {len(combined_df)}"]

    if debit_col:
        lines.append(
            f"Total Debits          : {combined_df[debit_col].dropna().sum():,.2f}")
        lines.append(
            f"Largest Debit         : {combined_df[debit_col].dropna().max():,.2f}")
        lines.append(
            f"Average Debit         : {combined_df[debit_col].dropna().mean():,.2f}")
        lines.append(
            f"Debit Count           : {combined_df[debit_col].dropna().count()}")

    if credit_col:
        lines.append(
            f"Total Credits         : {combined_df[credit_col].dropna().sum():,.2f}")
        lines.append(
            f"Largest Credit        : {combined_df[credit_col].dropna().max():,.2f}")
        lines.append(
            f"Average Credit        : {combined_df[credit_col].dropna().mean():,.2f}")
        lines.append(
            f"Credit Count          : {combined_df[credit_col].dropna().count()}")

    if balance_col:
        non_null = combined_df[balance_col].dropna()
        if len(non_null) > 0:
            lines.append(f"Opening Balance       : {non_null.iloc[0]:,.2f}")
            lines.append(f"Closing Balance       : {non_null.iloc[-1]:,.2f}")

    lines.append("=========================================")
    return "\n".join(lines)



#  BUILD THE RAG CHAIN


def build_rag_chain(vectorstore: PineconeVectorStore):


    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": 8}
        # k=8 → retrieve top 8 most similar chunks from Pinecone
        # More chunks = more context for the LLM
    )

    def format_docs(docs: list[Document]) -> str:

        formatted = []
        for i, doc in enumerate(docs):
            header = (f"--- Chunk {i+1} | "
                      f"Page {doc.metadata.get('page', '?')} | "
                      f"Type: {doc.metadata.get('chunk_type', '?')} ---")
            formatted.append(header + "\n" + doc.page_content)
        return "\n\n".join(formatted)

    # Prompt template for the LLM
    prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst AI assistant.
You are analyzing a bank or financial statement.

RULES:
1. Answer ONLY using the provided context and calculated results below
2. NEVER guess or make up any number
3. If calculated results are provided, use those exact numbers
4. Always mention dates, descriptions, and amounts clearly
5. If information is not found, say: "Not found in the statement"
6. Format amounts clearly: ₹1,25,000.00 or $1,250.00

---STATEMENT CONTEXT---
{context}

---CALCULATED RESULTS (use these for any math questions)---
{calculated}

---USER QUESTION---
{question}

---YOUR ANSWER---
""")

    llm = get_qwen_llm()

    chain = (
        {
            "context"   : retriever | format_docs,
            # retriever finds top 8 chunks → format_docs joins them as string

            "question"  : RunnablePassthrough(),
            # passes the question through unchanged

            "calculated": lambda _: ""
            # placeholder — gets replaced with real calculations in ask()
            # We set it to empty string here; the ask() function fills it in
        }
        | prompt
        | llm
        | StrOutputParser()

    )

    return chain, retriever




#  SMART QUESTION HANDLER


CALCULATION_KEYWORDS = [
    "total", "sum", "how much", "calculate", "average", "avg",
    "count", "how many", "maximum", "minimum", "largest", "smallest",
    "highest", "lowest", "balance", "closing", "opening", "spent",
    "received", "earned", "paid", "withdrawn", "deposited"
]
# If the user's question contains any of these words,
# we run the Python calculator BEFORE calling the LLM

def ask(question: str, chain, retriever, vectorstore) -> str:


    question_lower = question.lower()

    # Check if question needs calculation
    needs_calculation = any(
        keyword in question_lower
        for keyword in CALCULATION_KEYWORDS
    )

    # Retrieve relevant chunks from Pinecone
    retrieved_docs = retriever.invoke(question)
    # Converts question to vector → finds 8 most similar chunks in Pinecone

    #  Run calculator if needed
    calculated_result = ""
    if needs_calculation:
        print("🔢 Calculation detected — running Python calculator...")
        calculated_result = run_calculator(retrieved_docs)
        if calculated_result:
            print(calculated_result)  # show calculated numbers in terminal

    #  Format retrieved docs into context string
    formatted_context = "\n\n".join([
        f"--- Chunk {i+1} | Page {doc.metadata.get('page','?')} | "
        f"Type {doc.metadata.get('chunk_type','?')} ---\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])


    prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst AI assistant.
You are analyzing a bank or financial statement.

RULES:
1. Answer ONLY using the provided context and calculated results below
2. NEVER guess or make up any number — if calculated results exist, use them
3. Always mention dates, descriptions, and amounts clearly
4. If information is not found, say: "Not found in the statement"
5. Format amounts clearly: ₹1,25,000.00 or $1,250.00

---STATEMENT CONTEXT---
{context}

---CALCULATED RESULTS (use these exact numbers for any math questions)---
{calculated}

---USER QUESTION---
{question}

---YOUR ANSWER---
""")

    llm = get_qwen_llm()

    # Build a simple one-time chain for this question
    response_chain = prompt | llm | StrOutputParser()

    answer = response_chain.invoke({
        "context"   : formatted_context,
        "calculated": calculated_result if calculated_result
                      else "No calculations needed for this question.",
        "question"  : question
    })

    return answer



# MAIN PIPELINE


def process_financial_statement(pdf_path: str):
    """
    Full pipeline:
    PDF → extract → row-safe chunks → NVIDIA embed → Pinecone → RAG chain
    """
    print(f"\n Processing: {pdf_path}")
    print("=" * 60)

    print("\n[1/3] Extracting text and tables from PDF...")
    extracted = extract_pdf_content(pdf_path)

    print("\n[2/3] Creating transaction-safe chunks...")
    chunks = chunk_by_transactions(extracted, rows_per_chunk=15)

    print("\n[3/3] Embedding with NVIDIA + storing in Pinecone...")
    vectorstore = index_documents(chunks)

    print("\n Building RAG chain...")
    chain, retriever = build_rag_chain(vectorstore)

    print("\n" + "=" * 60)
    print("🚀 READY!")
    print("   Embedding : NVIDIA nemoretriever-300m (1024 dims)")
    print("   LLM       : Qwen3.5-122B-A10B (256K context)")
    print("   Vector DB : Pinecone")
    print("   Calculator: Python pandas (real math, no guessing)")
    print("=" * 60 + "\n")

    return chain, retriever, vectorstore



# INTERACTIVE Q&A


def ask_questions(chain, retriever, vectorstore):
    print("\n💬 Financial Statement Q&A")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    examples = [
        "What is the total debit amount?",
        "What is the total credit amount?",
        "What is the opening and closing balance?",
        "How many transactions are there?",
        "What is the largest withdrawal?",
        "What is the average transaction amount?",
        "Summarize this bank statement",
    ]
    print("Example questions:")
    for q in examples:
        print(f"  → {q}")
    print()

    while True:
        question = input("Your question: ").strip()

        if question.lower() == 'exit':
            print("\nGoodbye!")
            break

        if not question:
            continue

        print("\n🤔 Thinking...\n")

        try:
            answer = ask(question, chain, retriever, vectorstore)
            print(f"\n📊 Answer:\n{answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try rephrasing your question.")

        print("\n" + "-" * 60 + "\n")



# RUN


if __name__ == "__main__":

    PDF_PATH = r""


    chain, retriever, vectorstore = process_financial_statement(PDF_PATH)
    ask_questions(chain, retriever, vectorstore)

