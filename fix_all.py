"""
ONE-STEP FIX
Cleans documents and rebuilds vector store in one go
"""

import sys
from pathlib import Path

# Import custom modules
from document_processor import DocumentProcessor
from embedding_handler import EmbeddingHandler
from vector_store import VectorStore

# Configuration
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    print("\n" + "="*60)
    print("🔧 ONE-STEP FIX: Clean & Rebuild")
    print("="*60)
    
    # STEP 1: Clean documents
    print("\n[1/5] 🧹 Cleaning documents directory...")
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove problematic sample.txt
    old_sample = DOCUMENTS_DIR / "sample.txt"
    if old_sample.exists():
        print("      Removing corrupted sample.txt...")
        old_sample.unlink()
    
    # Create clean documents
    print("      Creating 4 clean sample documents...")
    
    docs = {
        "ai_overview.txt": """Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.

Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Deep Learning is a subset of machine learning based on artificial neural networks. The learning process is deep because the structure of neural networks consists of multiple input, output, and hidden layers.

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with machine learning and deep learning models.

Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos, machines can accurately identify and classify objects.
""",
        "machine_learning.txt": """Machine Learning Fundamentals

Machine Learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.

Supervised Learning uses labeled training data to learn the relationship between input and output. Common algorithms include linear regression, logistic regression, decision trees, and neural networks.

Unsupervised Learning works with unlabeled data to find hidden patterns or structures. Clustering and dimensionality reduction are common unsupervised learning techniques.

Reinforcement Learning is about taking suitable actions to maximize reward in a particular situation. The algorithm learns through trial and error, receiving rewards for correct actions.

Training Data is crucial for machine learning models. The quality and quantity of training data directly impacts the model performance and accuracy.
""",
        "ai_applications.txt": """AI Applications in the Real World

Healthcare: AI is revolutionizing healthcare through improved diagnostics, personalized treatment plans, drug discovery, and patient care. Medical imaging analysis using deep learning can detect diseases earlier and more accurately.

Finance: AI powers fraud detection systems, algorithmic trading, credit scoring, and risk assessment. Financial institutions use machine learning to analyze market trends and make investment decisions.

Transportation: Self-driving cars use computer vision, sensor fusion, and deep learning to navigate roads safely. AI optimizes traffic flow and predicts maintenance needs.

Customer Service: Chatbots and virtual assistants powered by NLP provide 24/7 customer support, handle routine queries, and escalate complex issues to human agents.

Education: AI enables personalized learning experiences, automated grading, intelligent tutoring systems, and adaptive learning platforms that adjust to individual student needs.
""",
        "python_basics.txt": """Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

Variables in Python are created when you assign a value to them. Python is dynamically typed, meaning you don't need to declare variable types explicitly.

Control Flow statements include if-else conditionals for decision making and loops like for and while for iteration. Python uses indentation to define code blocks.

Functions are reusable blocks of code that perform specific tasks. They are defined using the def keyword followed by the function name and parameters.

Lists are ordered, mutable collections that can contain items of different types. They are created using square brackets and support operations like append and insert.

Python has a rich ecosystem of libraries including NumPy for numerical computing, Pandas for data analysis, Flask and Django for web development, and TensorFlow for machine learning.
"""
    }
    
    for filename, content in docs.items():
        doc_path = DOCUMENTS_DIR / filename
        doc_path.write_text(content.strip(), encoding='utf-8')
        print(f"      ✅ {filename}")
    
    # STEP 2: Process documents
    print("\n[2/5] 📄 Processing documents...")
    document_processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = document_processor.process_directory(str(DOCUMENTS_DIR))
    
    if not chunks:
        print("      ❌ Failed to create chunks!")
        return False
    
    print(f"      ✅ Created {len(chunks)} chunks")
    
    # STEP 3: Load embedding model
    print("\n[3/5] 🧮 Loading embedding model...")
    embedding_handler = EmbeddingHandler(model_name=EMBEDDING_MODEL)
    embedding_handler.load_model()
    print(f"      ✅ Model loaded (dimension: {embedding_handler.get_embedding_dimension()})")
    
    # STEP 4: Generate embeddings
    print("\n[4/5] 🔢 Generating embeddings...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_handler.generate_embeddings(texts, show_progress=True)
    print(f"      ✅ Generated {len(embeddings)} embeddings")
    
    # STEP 5: Build and save vector store
    print("\n[5/5] 💾 Building vector store...")
    vector_store = VectorStore(
        embedding_dimension=embedding_handler.get_embedding_dimension(),
        index_path=str(VECTOR_DB_DIR)
    )
    vector_store.add_documents(embeddings, texts, chunks)
    vector_store.save()
    
    # Show final stats
    print("\n" + "="*60)
    print("✅ FIX COMPLETE!")
    print("="*60)
    stats = vector_store.get_stats()
    print(f"\n📊 Vector Store Stats:")
    print(f"   Documents indexed: {stats['total_documents']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Saved to: {stats['index_path']}")
    
    print("\n🚀 Next step: Run your Flask app")
    print("   python app.py")
    print("\n✨ Your chatbot is ready to use!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
