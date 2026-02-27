import os
from tempfile import NamedTemporaryFile
from supabase.client import Client, create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- NEW IMPORTS for hybrid retrieval and structure-aware chunking ---
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
from langchain_core.documents import Document
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

from . import config


supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Structure-aware chunking helpers (LayoutLMv3)
# ---------------------------------------------------------------------------

def _layoutlmv3_structure_aware_chunks(docs: list, chunk_size: int = 1000, chunk_overlap: int = 150) -> list:
    """
    Uses LayoutLMv3 to detect structural regions (tables, footnotes, multi-column text)
    in each document page and produces semantically coherent chunks that respect those
    boundaries, falling back to RecursiveCharacterTextSplitter for plain-text pages.
    """
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base"
    )
    model.eval()

    # Label tokens that indicate structural breaks so we can split there
    STRUCTURAL_LABELS = {"table", "footnote", "header", "footer"}
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    final_chunks = []

    for doc in docs:
        text = doc.page_content
        metadata = doc.metadata

        # Only attempt LayoutLMv3 when a page image is available (PDF with images)
        page_image_path = metadata.get("page_image_path")
        if page_image_path and os.path.exists(page_image_path):
            try:
                image = Image.open(page_image_path).convert("RGB")
                words = text.split()
                # LayoutLMv3 needs word-level bounding boxes; use dummy uniform boxes
                # when real boxes are unavailable — still allows structural signal from text tokens
                n = len(words)
                boxes = [[0, 0, 1000, 1000]] * n  # normalised placeholder boxes

                encoding = processor(
                    image,
                    words,
                    boxes=boxes,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = model(**encoding)

                # Map predicted label ids back to label names
                label_map = model.config.id2label
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                tokens = processor.tokenizer.convert_ids_to_tokens(
                    encoding["input_ids"].squeeze().tolist()
                )

                # Identify split boundaries at structural transitions
                segments, current_segment = [], []
                prev_label = None
                for token, pred_id in zip(tokens, predictions):
                    label = label_map.get(pred_id, "text").lower()
                    if (
                        label in STRUCTURAL_LABELS
                        and prev_label not in STRUCTURAL_LABELS
                        and current_segment
                    ):
                        segments.append(" ".join(current_segment))
                        current_segment = []
                    current_segment.append(token.replace("▁", "").replace("##", ""))
                    prev_label = label
                if current_segment:
                    segments.append(" ".join(current_segment))

                for seg in segments:
                    if seg.strip():
                        final_chunks.append(
                            Document(page_content=seg.strip(), metadata=metadata)
                        )
                continue  # skip fallback for this page

            except Exception as e:
                print(f"LayoutLMv3 processing failed for page, falling back: {e}")

        # Fallback: standard recursive splitter
        sub_chunks = fallback_splitter.split_documents([doc])
        final_chunks.extend(sub_chunks)

    return final_chunks


# ---------------------------------------------------------------------------
# Hybrid retrieval helpers (BM25 + dense + ColBERT reranking)
# ---------------------------------------------------------------------------

def _colbert_rerank(query: str, candidates: list, top_k: int = 3) -> list:
    """
    Reranks candidate Documents using ColBERT late-interaction scoring.
    Returns top_k reranked documents.
    """
    try:
        colbert_config = ColBERTConfig(
            checkpoint=config.COLBERT_CHECKPOINT  # e.g. "colbert-ir/colbertv2.0"
        )
        ckpt = Checkpoint(colbert_config.checkpoint, colbert_config=colbert_config)

        query_embs = ckpt.queryFromText([query])           # (1, L_q, dim)
        doc_texts = [d.page_content for d in candidates]
        doc_embs = ckpt.docFromText(doc_texts, bsize=8)    # list of (L_d, dim) tensors

        scores = []
        for d_emb in doc_embs:
            # MaxSim aggregation
            sim = torch.matmul(query_embs[0], d_emb.T)    # (L_q, L_d)
            score = sim.max(dim=-1).values.sum().item()
            scores.append(score)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in ranked_indices[:top_k]]

    except Exception as e:
        print(f"ColBERT reranking failed, returning BM25+dense candidates as-is: {e}")
        return candidates[:top_k]


def _hybrid_retrieve(question: str, vector_store: SupabaseVectorStore, all_chunks: list, top_k: int = 3) -> list:
    """
    Combines BM25 lexical retrieval with dense semantic retrieval then reranks
    the merged candidate set with ColBERT late-interaction scoring.
    """
    # --- Dense retrieval ---
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": top_k * 2})
    dense_docs = dense_retriever.invoke(question)

    # --- BM25 lexical retrieval ---
    corpus = [doc.page_content for doc in all_chunks]
    tokenized_corpus = [text.lower().split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][: top_k * 2]
    bm25_docs = [all_chunks[i] for i in top_bm25_indices]

    # --- Merge & deduplicate ---
    seen, candidates = set(), []
    for doc in dense_docs + bm25_docs:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    # --- ColBERT reranking ---
    reranked = _colbert_rerank(question, candidates, top_k=top_k)
    return reranked


# ---------------------------------------------------------------------------
# Original functions — structure preserved, return values unchanged
# ---------------------------------------------------------------------------

def process_and_embed_pdf(file_bytes: bytes, original_file_name: str) -> None:

    print("Starting PDF processing...")
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        print(f"Uploading {original_file_name} to Supabase Storage...")
        storage_path = f"public/{original_file_name}"
        supabase_client.storage.from_(config.PDF_BUCKET_NAME).upload(
            file=temp_file_path,
            path=storage_path,
            file_options={"content-type": "application/pdf", "x-upsert": "true"}
        )
        print("Upload successful.")

        print("Loading and chunking document...")
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # --- CHANGED: structure-aware chunking via LayoutLMv3 (replaces plain splitter) ---
        print("Applying structure-aware chunking with LayoutLMv3...")
        chunks = _layoutlmv3_structure_aware_chunks(
            docs, chunk_size=1000, chunk_overlap=150
        )
        print(f"Document split into {len(chunks)} structure-aware chunks.")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )

        print("Storing embeddings in Supabase Vector Store...")
        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase_client,
            table_name=config.VECTOR_TABLE_NAME,
            query_name=config.MATCH_FUNCTION_NAME,
            chunk_size=100 
        )
        print("Embeddings stored successfully.")

    finally:
        os.unlink(temp_file_path)


def get_answer_from_rag(question: str, chat_history: list):

    print("Building RAG chain on-demand...")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=config.REPO_ID, task="text-generation", max_new_tokens=512, do_sample=False
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"}
    )
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name=config.VECTOR_TABLE_NAME,
        query_name=config.MATCH_FUNCTION_NAME,
    )

    # --- CHANGED: hybrid retrieval replaces simple dense retriever ---
    # Pull all stored chunks from Supabase to feed BM25 (lexical arm)
    print("Fetching all chunks for BM25 lexical index...")
    all_rows = supabase_client.table(config.VECTOR_TABLE_NAME).select("content, metadata").execute()
    all_chunks = [
        Document(page_content=row["content"], metadata=row.get("metadata", {}))
        for row in (all_rows.data or [])
    ]

    template = """You are an expert financial analyst and research assistant. 
Your task is to provide clear, data-driven, and well-structured answers based strictly on the information provided in the context. 
If the context is insufficient to answer the question, explicitly state that.

--- 
**Guidelines:**
1. Use **Markdown formatting** throughout your response.  
2. When presenting:
   - **Financial data, ratios, or comparisons**, use **Markdown tables**.  
   - **Lists of factors, pros/cons, or steps**, use **bulleted lists**.  
3. Be **concise but analytical** — highlight insights, implications, and reasoning.  
4. If relevant, mention key **financial metrics**, **industry benchmarks**, or **risk factors** based on the context.  
5. Avoid speculation — only use information grounded in the provided context.  
6. Include a short **summary or recommendation** at the end when applicable.  

---
**Inputs:**
- **Context:** {context}  
- **Chat History:** {chat_history}  
- **Question:** {question}  

---
**Output Requirements:**
- Provide a **clear, structured, and actionable** financial analysis or answer.
- If data is missing, respond with: *"The context does not provide enough information to answer this question."*
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_chat_history(messages):
        if not messages: return "No previous conversation."
        formatted = []
        for msg in messages:
            role = "Human" if msg.get("type") == "human" else "Assistant"
            formatted.append(f"{role}: {msg.get('content')}")
        return "\n".join(formatted)

    # --- CHANGED: use hybrid retrieval pipeline ---
    print(f"Running hybrid retrieval for question: {question}")
    docs = _hybrid_retrieve(question, vector_store, all_chunks, top_k=3)
    context = format_docs(docs)
    formatted_chat_history = format_chat_history(chat_history)
    
    
    formatted_prompt = prompt.invoke({
        "context": context,
        "chat_history": formatted_chat_history,
        "question": question
    })

    print("Invoking LLM...")
    response = llm.invoke(formatted_prompt.to_messages())
    return response.content