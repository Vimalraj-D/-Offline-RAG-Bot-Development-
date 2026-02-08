"""
Main Flask Application
Improved Offline RAG Chatbot with Better Accuracy
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json

# Import custom modules
from document_processor import DocumentProcessor
from embedding_handler import EmbeddingHandler
from vector_store import VectorStore
from llm_handler import LLMHandler
from evaluate_rag import run_evaluation as run_rag_benchmark

# Configuration
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
MODELS_DIR = BASE_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
UPLOAD_FOLDER = DOCUMENTS_DIR
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'docx', 'csv'}

FALLBACK_ANSWER = "The provided documents do not contain enough information to answer this question."

# RAG settings: reduce context noise, stricter grounding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4
RETRIEVAL_K = 10
MIN_SIMILARITY = 0.20
MAX_TOKENS = 512
TEMPERATURE = 0.2
USE_QUERY_EXPANSION = True
USE_RERANK = True
MAX_CONTEXT_CHARS = 4000

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global components
document_processor = None
embedding_handler = None
vector_store = None
llm_handler = None
initialized = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_models():
    """Check if required models are downloaded"""
    # Check for LLM model
    llm_files = list(LLM_DIR.glob("*.gguf"))
    if not llm_files:
        print("\n❌ LLM model not found!")
        print("Please run: python download_models.py")
        return False
    
    return True


def initialize_system():
    """Initialize all RAG components with improved configuration"""
    global document_processor, embedding_handler, vector_store, llm_handler, initialized
    
    if initialized:
        return True
    
    print("\n" + "="*70)
    print("🚀 Initializing Improved Offline RAG System")
    print("="*70)
    
    # Check models
    if not check_models():
        return False
    
    try:
        # 1. Initialize document processor with optimized settings
        print("\n📄 Initializing document processor...")
        document_processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # 2. Initialize embedding handler
        print("\n🧮 Initializing embedding handler...")
        embedding_handler = EmbeddingHandler(model_name=EMBEDDING_MODEL)
        embedding_handler.load_model()
        
        # 3. Initialize vector store
        print("\n💾 Initializing vector store...")
        vector_store = VectorStore(
            embedding_dimension=embedding_handler.get_embedding_dimension(),
            index_path=str(VECTOR_DB_DIR)
        )
        
        # Try to load existing index
        if not vector_store.load():
            print("\n📚 Processing documents...")
            
            # Create documents directory if it doesn't exist
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Check if there are documents
            doc_files = (
                list(DOCUMENTS_DIR.glob("*.pdf")) + list(DOCUMENTS_DIR.glob("*.txt")) +
                list(DOCUMENTS_DIR.glob("*.md")) + list(DOCUMENTS_DIR.glob("*.docx")) +
                list(DOCUMENTS_DIR.glob("*.csv"))
            )
            
            if not doc_files:
                print(f"\n⚠️  No documents found in {DOCUMENTS_DIR}")
                print("Creating sample document for testing...")
                
                # Create comprehensive sample document
                sample_doc = DOCUMENTS_DIR / "ai_overview.txt"
                sample_doc.write_text("""
# Artificial Intelligence: A Comprehensive Overview

## Introduction to AI

Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines, 
especially computer systems. These processes include learning (the acquisition of information and 
rules for using the information), reasoning (using rules to reach approximate or definite conclusions), 
and self-correction.

## Core Concepts

### Machine Learning

Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn 
and improve from experience without being explicitly programmed. Machine learning focuses on the 
development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, 
or instruction, in order to look for patterns in data and make better decisions in the future. 
The primary aim is to allow computers to learn automatically without human intervention or 
assistance and adjust actions accordingly.

### Deep Learning

Deep Learning is a subset of machine learning that uses neural networks with multiple layers 
(hence "deep"). These neural networks attempt to simulate the behavior of the human brain, 
allowing it to "learn" from large amounts of data. While a neural network with a single layer 
can still make approximate predictions, additional hidden layers can help optimize the accuracy.

Deep learning drives many artificial intelligence applications and services that improve automation, 
performing analytical and physical tasks without human intervention. Deep learning technology lies 
behind everyday products and services such as digital assistants, voice-enabled TV remotes, and 
credit card fraud detection.

### Natural Language Processing

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, 
and manipulate human language. NLP draws from many disciplines, including computer science and 
computational linguistics, in its pursuit to fill the gap between human communication and computer 
understanding.

NLP enables computers to:
- Understand the meaning of text and speech
- Generate human-like text
- Translate between languages
- Answer questions
- Summarize documents
- Extract information from unstructured text

## Applications of AI

### Healthcare

In healthcare, AI is being used for:
- Medical diagnosis and treatment recommendations
- Drug discovery and development
- Personalized medicine
- Medical imaging analysis
- Patient monitoring and predictive analytics

AI systems can analyze complex medical data much faster than humans and identify patterns 
that might be missed by human doctors.

### Finance

The finance industry uses AI for:
- Algorithmic trading
- Fraud detection and prevention
- Credit scoring and loan approval
- Risk assessment
- Customer service chatbots
- Portfolio management

### Transportation

AI is revolutionizing transportation through:
- Self-driving cars and autonomous vehicles
- Traffic prediction and management
- Route optimization
- Predictive maintenance
- Smart parking systems

### Customer Service

AI-powered customer service includes:
- Chatbots and virtual assistants
- Sentiment analysis
- Automated email responses
- Voice recognition systems
- Personalized recommendations

## Ethical Considerations

As AI becomes more prevalent, several ethical concerns have emerged:

1. Privacy: AI systems often require large amounts of data, raising concerns about data privacy
2. Bias: AI systems can perpetuate or amplify existing biases present in training data
3. Transparency: Many AI systems operate as "black boxes," making it difficult to understand their decisions
4. Job displacement: Automation through AI may lead to job losses in certain sectors
5. Accountability: Determining responsibility when AI systems make mistakes or cause harm

## The Future of AI

The future of AI holds enormous potential:
- More advanced natural language understanding
- Improved computer vision and image recognition
- Better autonomous systems
- Enhanced personalization in all aspects of life
- Integration with other emerging technologies like quantum computing

However, realizing this potential will require addressing the ethical challenges and ensuring 
AI is developed and deployed responsibly for the benefit of all humanity.
                """, encoding='utf-8')
                print(f"   ✓ Sample document created: {sample_doc.name}")
            
            # Process documents
            chunks = document_processor.process_directory(str(DOCUMENTS_DIR))
            
            if chunks:
                # Generate embeddings
                print("\n🔢 Generating embeddings...")
                texts = [chunk['text'] for chunk in chunks]
                embeddings = embedding_handler.generate_embeddings(texts)
                
                # Add to vector store
                print("\n💾 Building vector index...")
                vector_store.add_documents(embeddings, texts, chunks)
                
                # Save index
                vector_store.save()
            else:
                print("\n⚠️  No content to process")
        
        # 4. Initialize LLM handler with optimized settings
        print("\n🤖 Initializing LLM...")
        llm_files = list(LLM_DIR.glob("*.gguf"))
        llm_handler = LLMHandler(
            model_path=str(llm_files[0]),
            n_ctx=2048,
            n_threads=4
        )
        llm_handler.load_model()
        
        initialized = True
        
        print("\n" + "="*70)
        print("✅ System initialized successfully!")
        print("="*70)
        
        stats = vector_store.get_stats()
        print(f"\n📊 System Stats:")
        print(f"   Documents indexed: {stats['total_documents']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Embedding model: {EMBEDDING_MODEL}")
        print(f"   LLM model: {llm_files[0].name}")
        print(f"   Chunk size: {CHUNK_SIZE} chars")
        print(f"   Chunk overlap: {CHUNK_OVERLAP} chars")
        print(f"   Top-K retrieval: {TOP_K}")
        print(f"\n🌐 Server ready at http://localhost:5000")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Serve the main page"""
    if not initialized:
        if not initialize_system():
            return "System initialization failed. Please check the console.", 500
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with improved RAG pipeline"""
    import time
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        t_start = time.perf_counter()
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}")
        
        # Query expansion for retrieval (broader recall; actual query used for rerank/LLM)
        retrieval_query = (query + " relevant information document context") if USE_QUERY_EXPANSION else query
        query_embedding = embedding_handler.generate_embedding(retrieval_query)
        exact_query_embedding = embedding_handler.generate_embedding(query) if USE_RERANK else None
        
        # Retrieve more candidates, then rerank for accuracy
        print(f"📚 Retrieving top-{TOP_K} relevant chunks (retrieval_k={RETRIEVAL_K})...")
        results = vector_store.search_with_reranking(
            query_embedding,
            top_k=RETRIEVAL_K,
            retrieval_k=RETRIEVAL_K
        )
        
        # Optional: rerank by exact query similarity for better precision
        if USE_RERANK and exact_query_embedding is not None and results:
            results = embedding_handler.rerank_by_similarity(exact_query_embedding, results, top_k=TOP_K)
        else:
            results = results[:TOP_K]
        
        results = [r for r in results if r['similarity'] >= MIN_SIMILARITY]

        if not results:
            t_total = time.perf_counter() - t_start
            return jsonify({
                'answer': FALLBACK_ANSWER,
                'sources': [],
                'chunks': [],
                'stats': {'total_time_sec': round(t_total, 2), 'chunks_used': 0, 'top_similarity': None}
            })
        
        total_context_len = sum(len(r['text']) for r in results)
        if total_context_len > MAX_CONTEXT_CHARS:
            trimmed = []
            acc = 0
            for r in results:
                if acc + len(r['text']) <= MAX_CONTEXT_CHARS:
                    trimmed.append(r)
                    acc += len(r['text'])
                else:
                    break
            results = trimmed if trimmed else results[:1]

        response = llm_handler.generate_answer(
            query,
            results,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        t_total = time.perf_counter() - t_start
        print(f"\n✅ Answer generated")
        print(f"   Sources used: {', '.join(response['sources'])}")
        print(f"{'='*70}\n")
        
        # Format chunks for display
        chunks_info = [
            {
                'text': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                'source': chunk['metadata'].get('source', 'Unknown'),
                'similarity': f"{chunk['similarity']:.1%}",
                'rank': chunk['rank']
            }
            for chunk in results
        ]
        
        return jsonify({
            'answer': response['answer'],
            'sources': response['sources'],
            'chunks': chunks_info,
            'stats': {
                'total_time_sec': round(t_total, 2),
                'chunks_used': len(results),
                'top_similarity': f"{results[0]['similarity']:.1%}" if results else None,
            }
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process new documents"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        skipped_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)
                
                filepath = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(str(filepath))
                uploaded_files.append(filename)
                print(f"✓ Uploaded: {filename}")
            else:
                skipped_files.append(file.filename)
        
        if not uploaded_files:
            return jsonify({
                'error': 'No valid files uploaded',
                'skipped': skipped_files
            }), 400
        
        print(f"\n📄 Processing {len(uploaded_files)} new document(s)...")
        
        # Process all documents
        chunks = document_processor.process_directory(str(DOCUMENTS_DIR))
        
        if chunks:
            # Generate embeddings
            print("🔢 Generating embeddings...")
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_handler.generate_embeddings(texts, show_progress=False)
            
            # Rebuild vector store
            print("💾 Rebuilding vector index...")
            vector_store.clear()
            vector_store.add_documents(embeddings, texts, chunks)
            vector_store.save()
            
            print(f"✅ Successfully indexed {len(chunks)} chunks from {len(uploaded_files)} document(s)")
            
            return jsonify({
                'message': 'Documents uploaded and indexed successfully',
                'uploaded': uploaded_files,
                'skipped': skipped_files,
                'total_chunks': len(chunks)
            })
        else:
            return jsonify({
                'error': 'Failed to process documents',
                'uploaded': uploaded_files,
                'skipped': skipped_files
            }), 500
            
    except Exception as e:
        print(f"❌ Error uploading documents: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        doc_files = (
            list(DOCUMENTS_DIR.glob("*.pdf")) + list(DOCUMENTS_DIR.glob("*.txt")) +
            list(DOCUMENTS_DIR.glob("*.md")) + list(DOCUMENTS_DIR.glob("*.docx")) +
            list(DOCUMENTS_DIR.glob("*.csv"))
        )
        
        documents = []
        for doc in sorted(doc_files):
            documents.append({
                'name': doc.name,
                'size': doc.stat().st_size,
                'modified': doc.stat().st_mtime
            })
        
        return jsonify({'documents': documents})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete a document and reindex"""
    try:
        from werkzeug.utils import secure_filename
        filename = secure_filename(filename)
        filepath = DOCUMENTS_DIR / filename
        
        if not filepath.exists():
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete file
        filepath.unlink()
        print(f"🗑️  Deleted: {filename}")
        
        # Reindex remaining documents
        chunks = document_processor.process_directory(str(DOCUMENTS_DIR))
        
        if chunks:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_handler.generate_embeddings(texts, show_progress=False)
            
            vector_store.clear()
            vector_store.add_documents(embeddings, texts, chunks)
            vector_store.save()
            print(f"✅ Reindexed {len(chunks)} chunks")
        else:
            # No documents left, clear vector store
            vector_store.clear()
            vector_store.save()
            print("📭 No documents remaining")
        
        return jsonify({
            'message': f'Document {filename} deleted successfully',
            'remaining_chunks': len(chunks) if chunks else 0
        })
        
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        vector_stats = vector_store.get_stats()
        return jsonify({
            'documents_indexed': vector_stats['total_documents'],
            'total_chunks': vector_stats.get('total_chunks', 0),
            'embedding_model': EMBEDDING_MODEL,
            'embedding_dimension': vector_stats['embedding_dimension'],
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/benchmark', methods=['GET'])
def benchmark():
    """Run RAG benchmark and return accuracy/latency metrics"""
    try:
        if not initialized:
            return jsonify({'error': 'System not initialized'}), 503
        report = run_rag_benchmark(
            embedding_handler=embedding_handler,
            vector_store=vector_store,
            llm_handler=llm_handler,
        )
        if report is None:
            return jsonify({'error': 'Benchmark failed (missing QA file or index)'}), 500
        return jsonify(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'initialized': initialized})


if __name__ == '__main__':
    # Fix Windows console encoding issues
    if sys.platform == 'win32':
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        import locale
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
            pass
    
    # Check if models need to be downloaded
    if not check_models():
        print("\n" + "="*70)
        print("📥 Models not found. Starting download...")
        print("="*70 + "\n")
        
        import download_models
        try:
            download_models.main()
        except SystemExit:
            print("\n❌ Model download failed. Please try again.")
            sys.exit(1)
    
    # Initialize system before starting server
    if not initialize_system():
        print("\n❌ Failed to initialize system. Exiting.")
        sys.exit(1)
    
    import logging
    from werkzeug.serving import run_simple
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    try:
        print("Server starting at http://localhost:5000")
    except OSError:
        pass
    run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=False)
