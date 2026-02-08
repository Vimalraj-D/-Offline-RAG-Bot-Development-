# Offline RAG Chatbot - Improved Version

A fully offline, privacy-focused Retrieval-Augmented Generation (RAG) chatbot with enhanced accuracy and professional UI.

## 🎯 Key Improvements

### RAG Accuracy Enhancements

1. **Better Prompt Engineering**
   - Optimized system prompts for more focused, accurate responses
   - Clear instructions to prevent hallucinations
   - Source citation requirements built into prompts
   - Lower temperature (0.3) for more deterministic outputs

2. **Improved Document Processing**
   - Increased chunk size (600 chars) for better context
   - Increased overlap (100 chars) for continuity
   - Smart chunking by paragraphs and sentences
   - Better text cleaning and normalization
   - Encoding detection for various file formats

3. **Enhanced Vector Search**
   - Uses Inner Product (IP) index for better cosine similarity
   - Proper embedding normalization
   - Search with reranking (retrieves 2x candidates, then reranks)
   - Minimum similarity threshold (0.3) to filter irrelevant results
   - Retrieves top-4 chunks instead of top-3

4. **Optimized Retrieval**
   - More context chunks (TOP_K=4)
   - Better similarity scoring
   - Relevance-based filtering
   - Enhanced metadata tracking

### UI/UX Improvements

1. **Professional Design**
   - Clean, minimal aesthetic with reduced border radius
   - No gradients - solid, professional colors
   - Consistent spacing and typography
   - Better readability with improved contrast
   - Modern, human-friendly interface

2. **Improved Visual Hierarchy**
   - Clear section separation
   - Better use of whitespace
   - Consistent iconography
   - Professional color palette (blue primary, clean grays)

3. **Enhanced Usability**
   - Smoother animations
   - Better loading states
   - Improved feedback messages
   - More intuitive file upload
   - Clearer error messages

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space for models
- CPU: 4+ cores recommended

## 🚀 Setup Instructions

### 1. Clone or Download

```bash
# If using git
git clone <repository-url>
cd offline-rag-bot

# Or simply extract the zip file
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
flask==3.0.0
llama-cpp-python==0.2.90
sentence-transformers==2.2.2
faiss-cpu==1.7.4
PyPDF2==3.0.1
numpy==1.24.3
tqdm==4.66.1
huggingface-hub==0.19.4
werkzeug==3.0.1
```

### 3. Download Models

```bash
python download_models.py
```

This will download:
- **Embedding Model**: all-MiniLM-L6-v2 (~80MB)
- **LLM Model**: Llama-3.2-1B-Instruct-Q4_K_M (~1.2GB)

### 4. Prepare Your Documents

Place your documents in the `data/documents/` directory:

```bash
mkdir -p data/documents
# Copy your PDFs, TXT, or MD files to data/documents/
```

Supported formats:
- PDF (`.pdf`)
- Text files (`.txt`)
- Markdown (`.md`)

### 5. Run the Application

```bash
python app.py
```

The server will:
1. Load models
2. Process documents
3. Build vector index
4. Start on `http://localhost:5000`

## 📁 Project Structure

```
offline-rag-bot/
├── app.py                      # Main Flask application (improved)
├── document_processor.py       # Document handling (improved chunking)
├── embedding_handler.py        # Embedding generation
├── vector_store.py            # FAISS vector database (improved search)
├── llm_handler.py             # LLM inference (improved prompting)
├── download_models.py         # Model downloader
├── templates/
│   └── index.html             # UI template (professional design)
├── static/
│   ├── css/
│   │   └── style.css          # Improved styling
│   └── js/
│       ├── app.js             # Main app logic
│       └── upload.js          # Upload functionality
├── data/
│   └── documents/             # Your documents go here
├── models/
│   ├── embeddings/            # Embedding model cache
│   └── llm/                   # LLM model files
└── vector_db/                 # FAISS index storage
```

## 🎨 UI Design Philosophy

The improved UI follows these principles:

1. **Minimalism**: Clean, distraction-free interface
2. **Consistency**: Uniform spacing, colors, and typography
3. **Clarity**: Clear visual hierarchy and information architecture
4. **Professionalism**: Business-appropriate design without flashy elements
5. **Accessibility**: Good contrast ratios and readable fonts

### Color Palette

- **Primary**: `#2563eb` (Professional blue)
- **Background**: `#ffffff` (Clean white)
- **Surface**: `#f8fafc` (Subtle gray)
- **Text Primary**: `#0f172a` (Dark slate)
- **Text Secondary**: `#475569` (Medium gray)
- **Border**: `#e2e8f0` (Light gray)

## 🔧 Configuration

Edit these settings in `app.py`:

```python
# RAG Settings
CHUNK_SIZE = 600           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
TOP_K = 4                  # Number of chunks to retrieve
MIN_SIMILARITY = 0.3       # Minimum similarity threshold
MAX_TOKENS = 768          # Max tokens in response
TEMPERATURE = 0.3          # LLM temperature (lower = more focused)
```

## 💡 Usage Tips

1. **Upload Quality Documents**
   - Use well-formatted PDFs or text files
   - Ensure text is readable (not scanned images)
   - Organize content logically

2. **Ask Better Questions**
   - Be specific in your queries
   - Reference document topics when possible
   - Ask one question at a time

3. **Verify Sources**
   - Always check the cited sources
   - Cross-reference important information
   - Use the context chunks to verify accuracy

4. **Document Management**
   - Remove outdated documents
   - Keep document collection focused
   - Re-upload when updating content

## 🔍 How It Works

### RAG Pipeline

1. **Document Processing**
   - Documents are loaded and cleaned
   - Text is split into overlapping chunks
   - Metadata is attached to each chunk

2. **Embedding Generation**
   - Each chunk is converted to a 384-dimensional vector
   - Vectors capture semantic meaning
   - Embeddings are normalized for cosine similarity

3. **Vector Storage**
   - Embeddings are stored in FAISS index
   - Fast similarity search (< 1ms for 1000s of docs)
   - Efficient retrieval of relevant chunks

4. **Query Processing**
   - User query is embedded using same model
   - Top-K most similar chunks are retrieved
   - Results are filtered by similarity threshold

5. **Answer Generation**
   - Retrieved chunks provide context
   - LLM generates answer based only on context
   - Sources are cited automatically

### Improved Accuracy Techniques

1. **Semantic Chunking**: Chunks respect paragraph and sentence boundaries
2. **Context Overlap**: Chunks share context for continuity
3. **Reranking**: Initial over-retrieval followed by reranking
4. **Threshold Filtering**: Only high-confidence matches are used
5. **Optimized Prompting**: Clear instructions prevent hallucination
6. **Lower Temperature**: More deterministic, focused outputs

## 🐛 Troubleshooting

### Models Not Downloading
```bash
# Manually download from:
# https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF
# Place .gguf file in models/llm/
```

### Out of Memory
- Reduce `TOP_K` to 2-3
- Use smaller chunk size (400-500)
- Close other applications
- Consider using a smaller LLM model

### Slow Responses
- Reduce `MAX_TOKENS` to 512
- Use fewer chunks (TOP_K = 2-3)
- Reduce `n_threads` in llm_handler.py

### Poor Accuracy
- Increase `CHUNK_SIZE` to 800
- Increase `TOP_K` to 5-6
- Lower `MIN_SIMILARITY` to 0.2
- Check document quality and formatting

## 📊 Performance

Typical performance on modern hardware:

- **Document Processing**: ~2-5 chunks/second
- **Embedding Generation**: ~100 chunks/second
- **Vector Search**: < 1ms for 1000s of documents
- **Answer Generation**: ~20-40 tokens/second (CPU)

## 🔒 Privacy

This is a fully offline system:

- ✅ No internet connection required after setup
- ✅ All processing happens locally
- ✅ No data sent to external servers
- ✅ Complete privacy and data control

## 📝 License

This project is provided as-is for educational and personal use.

## 🤝 Contributing

Feel free to submit issues or improvements!

## 🙏 Acknowledgments

- **SentenceTransformers** for embedding models
- **FAISS** for efficient vector search
- **Llama.cpp** for LLM inference
- **Meta** for Llama models
- **HuggingFace** for model hosting

---

**Built with ❤️ for privacy-focused AI applications**
