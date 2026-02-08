"""
LLM Handler
Local LLM inference with strict context-only RAG prompt and mandatory fallback.
"""

from pathlib import Path
from typing import List, Dict

from llama_cpp import Llama

FALLBACK_ANSWER = "The provided documents do not contain enough information to answer this question."


class LLMHandler:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}. Run: python download_models.py")
        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )

    def create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_parts = []
        for i, ch in enumerate(context_chunks, 1):
            src = ch["metadata"].get("source", "Unknown")
            context_parts.append(f"[{i}] Source: {src}\n{ch['text'].strip()}")
        context = "\n\n".join(context_parts)

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You answer using ONLY the CONTEXT below. No external knowledge. No hallucination. If the answer is not in CONTEXT, reply with exactly: {FALLBACK_ANSWER}

When CONTEXT is relevant, you MUST respond in this structured format. Use short lines (no long paragraphs).

RESPONSE FORMAT (use these section headers):
## Title
(One short title for the answer.)

## Key Summary
- Bullet 1
- Bullet 2
- Bullet 3 (if needed)

## Detailed Breakdown
(Use bullets, or numbered steps for procedures, or a markdown table when comparing values. Use **bold** for key facts. Keep each line short.)

## Source Evidence
(1–2 exact sentences or phrases from CONTEXT that support the answer. Use quotes.)

## Missing Information
(Only include this section if something was asked but not found in CONTEXT. Otherwise omit.)

FORMATTING RULES:
- Use ## before every section header.
- Use bullet points (- or *) instead of long paragraphs.
- Use tables (| Col1 | Col2 |) when listing or comparing attributes.
- Use numbered steps (1. 2. 3.) for procedures.
- Use **bold** for key facts only.
- Use code blocks only if the user asked for code.
- Keep line length short; no wall-of-text.<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

QUESTION: {query}

Answer in the structured format above using ONLY the CONTEXT. If CONTEXT has no relevant information, reply with exactly: {FALLBACK_ANSWER}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Dict:
        if self.model is None:
            self.load_model()
        prompt = self.create_prompt(query, context_chunks)
        try:
            out = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.1,
                stop=["<|eot_id|>", "<|end_of_text|>", "\n\nQUESTION:", "\n\nCONTEXT:"],
                echo=False,
            )
            answer = (out["choices"][0].get("text") or "").strip()
            answer = self._normalize_answer(answer)
        except Exception as e:
            print(f"LLM error: {e}")
            answer = FALLBACK_ANSWER
        sources = list({ch["metadata"].get("source", "Unknown") for ch in context_chunks})
        return {
            "answer": answer,
            "sources": sources,
            "context_chunks": len(context_chunks),
            "query": query,
            "relevance_scores": [ch.get("similarity", 0) for ch in context_chunks],
        }

    def _normalize_answer(self, answer: str) -> str:
        answer = answer.strip()
        if not answer:
            return FALLBACK_ANSWER
        lower = answer.lower()
        first_120 = lower[:120]
        refusal_starts = (
            "the provided documents do not contain",
            "i don't have enough",
            "i do not have enough",
            "there is no information in",
            "the context does not contain",
            "the documents do not contain",
        )
        if any(first_120.startswith(s) for s in refusal_starts):
            return FALLBACK_ANSWER
        return answer

    def generate_streaming(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        if self.model is None:
            self.load_model()
        prompt = self.create_prompt(query, context_chunks)
        stream = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.1,
            stream=True,
            stop=["<|eot_id|>", "<|end_of_text|>", "\n\nQUESTION:", "\n\nCONTEXT:"],
        )
        for item in stream:
            yield item["choices"][0].get("text", "")
