"""
Document Processor
Text extraction from PDF, TXT, MD, DOCX, CSV and chunking with overlap.
"""

import csv
import re
from pathlib import Path
from typing import List, Dict

import PyPDF2

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


class DocumentProcessor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, filepath: str) -> str:
        text = ""
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        text += f"\n[Page {i + 1}]\n{t}\n"
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
        return text

    def load_text(self, filepath: str) -> str:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, OSError):
                continue
        print(f"Could not decode {filepath}")
        return ""

    def load_docx(self, filepath: str) -> str:
        if DocxDocument is None:
            print("python-docx not installed; cannot read DOCX")
            return ""
        try:
            doc = DocxDocument(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"Error reading DOCX {filepath}: {e}")
            return ""

    def load_csv(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = []
                for row in reader:
                    rows.append(" | ".join(cell.strip() for cell in row))
                return "\n".join(rows)
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1", newline="") as f:
                reader = csv.reader(f)
                rows = [" | ".join(cell.strip() for cell in row) for row in reader]
                return "\n".join(rows)
        except Exception as e:
            print(f"Error reading CSV {filepath}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"-\n", "", text)
        text = "".join(c for c in text if c.isprintable() or c in "\n\t")
        return text.strip()

    def create_chunks(self, text: str, source: str = "") -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                if len(p) > self.chunk_size:
                    for sent in self._split_sentences(p):
                        if len(current) + len(sent) + 1 <= self.chunk_size:
                            current += sent + " "
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = sent + " "
                else:
                    current = p
            else:
                current = (current + "\n\n" + p) if current else p
        if current:
            chunks.append(current.strip())
        overlapped = self._add_overlap(chunks)
        return [c for c in overlapped if len(c) > 50]

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            if "." in overlap:
                overlap = overlap[overlap.rfind(".") + 1 :].strip()
            out.append((overlap + " " + chunks[i]) if overlap else chunks[i])
        return out

    def process_document(self, filepath: str) -> List[Dict]:
        path = Path(filepath)
        ext = path.suffix.lower()
        if ext == ".pdf":
            text = self.load_pdf(str(path))
        elif ext == ".docx":
            text = self.load_docx(str(path))
        elif ext == ".csv":
            text = self.load_csv(str(path))
        else:
            text = self.load_text(str(path))
        if not text:
            return []
        text = self.clean_text(text)
        raw_chunks = self.create_chunks(text, path.name)
        return [
            {
                "text": c,
                "source": path.name,
                "chunk_id": i,
                "total_chunks": len(raw_chunks),
                "chunk_size": len(c),
                "file_type": ext,
            }
            for i, c in enumerate(raw_chunks)
        ]

    def process_directory(self, directory: str) -> List[Dict]:
        root = Path(directory)
        if not root.exists():
            return []
        patterns = ["*.pdf", "*.txt", "*.md", "*.docx", "*.csv"]
        files = []
        for p in patterns:
            files.extend(root.glob(p))
        files = sorted(set(files))
        if not files:
            return []
        all_chunks = []
        for fp in files:
            chunks = self.process_document(fp)
            all_chunks.extend(chunks)
        return all_chunks
