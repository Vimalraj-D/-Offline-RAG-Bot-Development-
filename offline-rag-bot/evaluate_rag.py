"""
RAG Benchmark / Evaluation Script
Measures accuracy and latency for reproducible, high benchmark results.
Run: python evaluate_rag.py (after app components are available)
"""

import json
import re
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from document_processor import DocumentProcessor
from embedding_handler import EmbeddingHandler
from vector_store import VectorStore
from llm_handler import LLMHandler

# Config (match app.py)
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
MODELS_DIR = BASE_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4
RETRIEVAL_K = 10
MIN_SIMILARITY = 0.35
MAX_TOKENS = 512
TEMPERATURE = 0.2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
USE_QUERY_EXPANSION = True
USE_RERANK = True


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def tokenize(s: str) -> set:
    return set(re.findall(r'\b[a-z0-9]+\b', normalize_text(s)))


def token_f1(pred: str, gold_keywords: list) -> float:
    """F1 based on overlap with expected keywords (precision/recall over tokens)."""
    pred_tokens = tokenize(pred)
    gold_tokens = set(normalize_text(w) for w in gold_keywords)
    if not gold_tokens:
        return 1.0
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_recall(pred: str, gold_keywords: list) -> float:
    """Fraction of expected keywords that appear in the answer."""
    pred_norm = normalize_text(pred)
    gold = [normalize_text(w) for w in gold_keywords]
    if not gold:
        return 1.0
    found = sum(1 for w in gold if w in pred_norm)
    return found / len(gold)


def contains_phrase(pred: str, phrases: list) -> bool:
    pred_norm = normalize_text(pred)
    return any(normalize_text(p) in pred_norm for p in phrases)


def run_evaluation(embedding_handler=None, vector_store=None, llm_handler=None):
    """
    Run RAG benchmark. If embedding_handler, vector_store, llm_handler are provided
    (e.g. from app), use them; otherwise load from disk.
    """
    qa_path = BASE_DIR / "data" / "benchmark_qa.json"
    if not qa_path.exists():
        print(f"Benchmark QA file not found: {qa_path}")
        return None

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    if embedding_handler is None or vector_store is None or llm_handler is None:
        print("Loading models and vector store...")
        doc_processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        embedding_handler = embedding_handler or EmbeddingHandler(model_name=EMBEDDING_MODEL)
        if embedding_handler.model is None:
            embedding_handler.load_model()
        if vector_store is None:
            vector_store = VectorStore(
                embedding_dimension=embedding_handler.get_embedding_dimension(),
                index_path=str(VECTOR_DB_DIR),
            )
            if not vector_store.load():
                print("Vector store not found. Index documents first (run app once).")
                return None
        if llm_handler is None:
            llm_files = list(LLM_DIR.glob("*.gguf"))
            if not llm_files:
                print("LLM model not found. Run download_models.py first.")
                return None
            llm_handler = LLMHandler(model_path=str(llm_files[0]), n_ctx=2048, n_threads=4)
            llm_handler.load_model()

    results = []
    total_time = 0
    total_f1 = 0
    total_keyword_recall = 0
    exact_phrase_hits = 0

    for i, item in enumerate(qa_pairs):
        q = item["question"]
        expected_kw = item.get("expected_keywords", [])
        expected_phrases = item.get("expected_answer_contains", [])

        t0 = time.perf_counter()
        # Retrieve
        retrieval_query = (q + " relevant information document context") if USE_QUERY_EXPANSION else q
        query_emb = embedding_handler.generate_embedding(retrieval_query)
        exact_emb = embedding_handler.generate_embedding(q) if USE_RERANK else None
        candidates = vector_store.search_with_reranking(query_emb, top_k=RETRIEVAL_K, retrieval_k=RETRIEVAL_K)
        if USE_RERANK and exact_emb is not None and candidates:
            candidates = embedding_handler.rerank_by_similarity(exact_emb, candidates, top_k=TOP_K)
        else:
            candidates = candidates[:TOP_K]
        candidates = [r for r in candidates if r["similarity"] >= MIN_SIMILARITY]

        if not candidates:
            pred = "I don't have any relevant information in the documents to answer this question."
        else:
            resp = llm_handler.generate_answer(q, candidates, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            pred = resp["answer"]

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        f1 = token_f1(pred, expected_kw)
        k_recall = keyword_recall(pred, expected_kw)
        phrase_ok = 1 if (not expected_phrases or contains_phrase(pred, expected_phrases)) else 0

        total_f1 += f1
        total_keyword_recall += k_recall
        exact_phrase_hits += phrase_ok

        results.append({
            "question": q,
            "answer_preview": (pred[:150] + "...") if len(pred) > 150 else pred,
            "token_f1": round(f1, 4),
            "keyword_recall": round(k_recall, 4),
            "phrase_match": bool(phrase_ok),
            "latency_sec": round(elapsed, 2),
        })

    n = len(qa_pairs)
    avg_latency = total_time / n if n else 0
    avg_f1 = total_f1 / n if n else 0
    avg_keyword_recall = total_keyword_recall / n if n else 0
    phrase_accuracy = exact_phrase_hits / n if n else 0

    report = {
        "summary": {
            "num_questions": n,
            "avg_token_f1": round(avg_f1, 4),
            "avg_keyword_recall": round(avg_keyword_recall, 4),
            "phrase_accuracy": round(phrase_accuracy, 4),
            "avg_latency_sec": round(avg_latency, 2),
            "total_time_sec": round(total_time, 2),
        },
        "results": results,
    }
    return report


def main():
    report = run_evaluation()
    if report is None:
        sys.exit(1)
    print("\n" + "=" * 60)
    print("RAG BENCHMARK RESULTS")
    print("=" * 60)
    s = report["summary"]
    print(f"  Questions:        {s['num_questions']}")
    print(f"  Avg Token F1:     {s['avg_token_f1']:.2%}")
    print(f"  Keyword Recall:   {s['avg_keyword_recall']:.2%}")
    print(f"  Phrase Accuracy:  {s['phrase_accuracy']:.2%}")
    print(f"  Avg Latency:      {s['avg_latency_sec']}s")
    print("=" * 60)
    out_path = BASE_DIR / "data" / "benchmark_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {out_path}")
    return report


if __name__ == "__main__":
    main()
