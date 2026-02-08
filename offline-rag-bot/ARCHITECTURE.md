# Architecture Documentation

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB BROWSER                              │
│                  (User Interface)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP Requests
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   FLASK WEB SERVER                           │
│                    (app.py)                                  │
│                                                              │
│  Routes:                                                     │
│  • GET  /           → Serve UI                              │
│  • POST /api/chat   → Process queries                       │
│  • GET  /api/stats  → System statistics                     │
│  • GET  /health     → Health check                          │
└───┬──────────────┬──────────────┬─────────────┬────────────┘
    │              │              │             │
    ▼              ▼              ▼             ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Document │  │Embedding │  │ Vector   │  │   LLM    │
│Processor│  │ Handler  │  │  Store   │  │ Handler  │
└─────────┘  └──────────┘  └──────────┘  └──────────┘
    │              │              │             │
    ▼              ▼              ▼             ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│PyPDF2   │  │Sentence  │  │  FAISS   │  │llama.cpp │
│         │  │Transform │  │          │  │          │
└─────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Component Details

### 1. Document Processor (`document_processor.py`)

**Purpose**: Ingest and preprocess documents

**Functions**:
- `load_pdf()`: Extract text from PDF files
- `load_text()`: Load plain text files
- `clean_text()`: Normalize and clean extracted text
- `create_chunks()`: Split text into overlapping chunks
- `process_document()`: Complete document processing pipeline
- `process_directory()`: Batch process multiple documents

**Data Flow**:
```
PDF/Text File → Load → Clean → Chunk → Metadata → Output
```

**Chunking Strategy**:
- Default chunk size: 500 characters
- Overlap: 50 characters
- Smart splitting at sentence boundaries
- Preserves context between chunks

### 2. Embedding Handler (`embedding_handler.py`)

**Purpose**: Generate vector embeddings for text

**Model**: `all-MiniLM-L6-v2`
- Dimension: 384
- Size: ~80MB
- Speed: ~500 sentences/second (CPU)
- Quality: Good for semantic search

**Functions**:
- `load_model()`: Initialize SentenceTransformer model
- `generate_embedding()`: Single text embedding
- `generate_embeddings()`: Batch embedding generation
- `compute_similarity()`: Cosine similarity calculation

**Data Flow**:
```
Text → Tokenize → Encode → 384-dim Vector
```

### 3. Vector Store (`vector_store.py`)

**Purpose**: Store and retrieve document embeddings

**Technology**: Facebook AI Similarity Search (FAISS)
- Index type: `IndexFlatL2` (exact search)
- Distance metric: L2 (Euclidean)
- Normalized for cosine similarity

**Functions**:
- `add_documents()`: Add embeddings to index
- `search()`: Find k-nearest neighbors
- `save()`: Persist index to disk
- `load()`: Load index from disk

**Search Process**:
```
Query Embedding → Normalize → FAISS Search → Top-K Results → Ranked by Similarity
```

**Similarity Scoring**:
```python
similarity = 1 - (L2_distance / 2)  # Range: [0, 1]
```

### 4. LLM Handler (`llm_handler.py`)

**Purpose**: Generate answers using local LLM

**Model**: Llama 3.2 1B Instruct (GGUF Q4_K_M)
- Size: ~1.2GB
- Context window: 2048 tokens
- Quantization: 4-bit (Q4_K_M)
- Format: GGUF (llama.cpp)

**Functions**:
- `load_model()`: Initialize llama.cpp model
- `create_prompt()`: Format prompt with context
- `generate_answer()`: Generate complete response
- `generate_streaming()`: Stream tokens (for real-time UI)

**Prompt Template**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant...
<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: [Retrieved chunks]
Question: [User query]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### 5. Flask Application (`app.py`)

**Purpose**: Web server and orchestration

**Key Routes**:

1. `GET /` → Serve UI
2. `POST /api/chat` → Process chat request
   - Generate query embedding
   - Retrieve relevant chunks
   - Generate LLM response
   - Return formatted answer
3. `GET /api/stats` → System statistics
4. `GET /health` → Health check

**Initialization Sequence**:
```
1. Check models exist
2. Load document processor
3. Load embedding model
4. Initialize vector store
5. Process documents (if needed)
6. Load LLM model
7. Start Flask server
```

## Data Flow

### Complete Query Processing Flow

```
1. User enters query in UI
   ↓
2. Frontend sends POST to /api/chat
   ↓
3. Embedding Handler converts query to vector
   ↓
4. Vector Store searches FAISS index
   ↓
5. Top-K most similar chunks retrieved
   ↓
6. LLM Handler formats prompt with chunks
   ↓
7. Llama model generates answer
   ↓
8. Response sent to frontend
   ↓
9. UI displays answer with sources
```

### Document Indexing Flow

```
1. Documents placed in data/documents/
   ↓
2. Document Processor loads and chunks files
   ↓
3. Embedding Handler generates embeddings
   ↓
4. Vector Store adds to FAISS index
   ↓
5. Index saved to vector_db/
```

## Storage Structure

```
offline-rag-bot/
├── data/
│   └── documents/          # User documents
│       ├── *.pdf
│       ├── *.txt
│       └── *.md
│
├── models/
│   ├── embeddings/         # Cached SentenceTransformers
│   │   └── [model cache]
│   └── llm/                # GGUF model files
│       └── *.gguf
│
├── vector_db/              # FAISS index
│   ├── index.faiss         # Vector index
│   └── index_metadata.pkl  # Chunk metadata
│
└── static/                 # Web assets
    ├── css/
    └── js/
```

## Performance Characteristics

### Memory Usage

| Component | RAM Usage |
|-----------|-----------|
| Base Python | ~100MB |
| SentenceTransformers | ~400MB |
| FAISS Index | ~10MB per 10K docs |
| Llama 3.2 1B | ~2GB |
| **Total** | **~2.5-3GB** |

### Processing Speed (CPU)

| Operation | Time |
|-----------|------|
| Document chunking | 1-2 sec/page |
| Embedding generation | 50-100ms per chunk |
| FAISS search | <10ms |
| LLM generation | 2-5 sec per response |
| **Total query time** | **2-6 seconds** |

### Disk Usage

| Component | Size |
|-----------|------|
| Code | ~50KB |
| Dependencies | ~500MB |
| Embedding model | ~80MB |
| LLM model | ~1.2GB |
| Vector index | ~1MB per 1000 docs |
| **Total** | **~2GB + documents** |

## Scalability

### Document Limits

- **Small (<1000 docs)**: Instant search, <100MB RAM for index
- **Medium (1000-10K docs)**: <1 sec search, ~1GB RAM
- **Large (10K-100K docs)**: 1-2 sec search, ~10GB RAM
- **Very Large (>100K docs)**: Consider FAISS approximate search

### Optimization Options

1. **Faster embeddings**: Use GPU-accelerated models
2. **Approximate search**: Switch to `IndexIVFFlat` for large collections
3. **Smaller LLM**: Use Phi-2 (2.7B) for faster inference
4. **Larger LLM**: Use Llama-7B for better quality (requires more RAM)

## Security & Privacy

### Data Privacy
- ✅ All processing happens locally
- ✅ No external API calls
- ✅ No telemetry or tracking
- ✅ Documents never leave your machine

### Network Isolation
- Only listens on localhost by default
- No outbound connections during runtime
- Models downloaded once, cached locally

## Extension Points

### Adding New Document Types

Modify `document_processor.py`:
```python
def load_docx(self, filepath):
    # Add DOCX support
    pass
```

### Custom Embedding Models

Modify `app.py`:
```python
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better quality
```

### Different LLMs

Replace GGUF file in `models/llm/` and update path in `app.py`

### API Extensions

Add new routes in `app.py`:
```python
@app.route('/api/documents', methods=['GET'])
def list_documents():
    # Return list of indexed documents
    pass
```

## Monitoring & Debugging

### Logging

Add to `app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

```python
import time

start = time.time()
# ... operation ...
print(f"Took {time.time() - start:.2f}s")
```

### Health Checks

- `/health` endpoint returns system status
- Frontend polls every 5 seconds
- Status indicator shows: Ready / Error / Initializing

## Troubleshooting Guide

### Common Issues

1. **Out of Memory**
   - Reduce `MAX_TOKENS`
   - Use smaller LLM model
   - Reduce `TOP_K`

2. **Slow Responses**
   - Reduce chunk count
   - Use smaller documents
   - Enable GPU acceleration

3. **Poor Answers**
   - Increase `TOP_K`
   - Improve chunking strategy
   - Use better embedding model
   - Use larger LLM

4. **Model Download Fails**
   - Check internet connection
   - Retry: `python download_models.py`
   - Manual download from Hugging Face

---

Built with open-source technologies for complete offline operation.
