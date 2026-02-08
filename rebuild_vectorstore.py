"""
Rebuild Vector Store
This script manually processes documents and rebuilds the vector database
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

# Settings (match app.py)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

def create_sample_documents():
    """Create sample documents for testing"""
    print("\n📝 Creating sample documents...")
    
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sample document 1: AI Overview
    doc1 = DOCUMENTS_DIR / "ai_overview.txt"
    doc1.write_text("""
Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. 
These processes include learning, reasoning, and self-correction. AI has become increasingly important in modern technology.

Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly 
programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Deep Learning is a subset of machine learning based on artificial neural networks. The learning process is deep because 
the structure of neural networks consists of multiple input, output, and hidden layers.

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human 
language. NLP combines computational linguistics with machine learning and deep learning models.

Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital 
images from cameras and videos and deep learning models, machines can accurately identify and classify objects.
""", encoding='utf-8')
    
    # Sample document 2: Machine Learning
    doc2 = DOCUMENTS_DIR / "machine_learning.txt"
    doc2.write_text("""
Machine Learning Fundamentals

Machine Learning is a method of data analysis that automates analytical model building. It is based on the idea that 
systems can learn from data, identify patterns, and make decisions with minimal human intervention.

Supervised Learning uses labeled training data to learn the relationship between input and output. Common algorithms 
include linear regression, logistic regression, decision trees, and neural networks.

Unsupervised Learning works with unlabeled data to find hidden patterns or structures. Clustering and dimensionality 
reduction are common unsupervised learning techniques.

Reinforcement Learning is about taking suitable actions to maximize reward in a particular situation. The algorithm 
learns through trial and error, receiving rewards for correct actions and penalties for incorrect ones.

Training Data is crucial for machine learning models. The quality and quantity of training data directly impacts the 
model's performance and accuracy.
""", encoding='utf-8')
    
    # Sample document 3: Applications
    doc3 = DOCUMENTS_DIR / "ai_applications.txt"
    doc3.write_text("""
AI Applications in Real World

Healthcare: AI is revolutionizing healthcare through improved diagnostics, personalized treatment plans, drug discovery, 
and patient care. Medical imaging analysis using deep learning can detect diseases earlier and more accurately.

Finance: AI powers fraud detection systems, algorithmic trading, credit scoring, and risk assessment. Financial 
institutions use machine learning to analyze market trends and make investment decisions.

Transportation: Self-driving cars use computer vision, sensor fusion, and deep learning to navigate roads safely. 
AI optimizes traffic flow, predicts maintenance needs, and improves logistics.

Customer Service: Chatbots and virtual assistants powered by NLP provide 24/7 customer support, handle routine queries, 
and escalate complex issues to human agents.

Education: AI enables personalized learning experiences, automated grading, intelligent tutoring systems, and adaptive 
learning platforms that adjust to individual student needs.

Manufacturing: AI improves quality control, predictive maintenance, supply chain optimization, and robotic automation 
in manufacturing processes.
""", encoding='utf-8')
    
    print(f"✅ Created {len(list(DOCUMENTS_DIR.glob('*.txt')))} sample documents")
    return list(DOCUMENTS_DIR.glob('*.txt'))

def rebuild_vector_store():
    """Rebuild the vector store from scratch"""
    print("\n" + "="*60)
    print("🔄 Rebuilding Vector Store")
    print("="*60)
    
    try:
        # 1. Create sample documents if needed
        doc_files = (
            list(DOCUMENTS_DIR.glob("*.pdf")) + list(DOCUMENTS_DIR.glob("*.txt")) +
            list(DOCUMENTS_DIR.glob("*.md")) + list(DOCUMENTS_DIR.glob("*.docx")) +
            list(DOCUMENTS_DIR.glob("*.csv"))
        )
        
        if not doc_files or len(doc_files) == 0:
            print("\n⚠️  No documents found. Creating samples...")
            create_sample_documents()
            doc_files = list(DOCUMENTS_DIR.glob("*.txt"))
        else:
            print(f"\n📚 Found {len(doc_files)} document(s)")
            for doc in doc_files:
                print(f"   - {doc.name}")
        
        # 2. Initialize document processor
        print("\n📄 Initializing document processor...")
        document_processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # 3. Process documents
        print("\n🔨 Processing documents...")
        chunks = document_processor.process_directory(str(DOCUMENTS_DIR))
        
        if not chunks:
            print("❌ No chunks were created!")
            print("\nDebugging information:")
            for doc_file in doc_files:
                print(f"\n   Trying to read: {doc_file}")
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"   ✅ Successfully read {len(content)} characters")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            return False
        
        print(f"\n✅ Created {len(chunks)} chunks from documents")
        
        # 4. Initialize embedding handler
        print("\n🧮 Initializing embedding handler...")
        embedding_handler = EmbeddingHandler(model_name=EMBEDDING_MODEL)
        embedding_handler.load_model()
        
        # 5. Generate embeddings
        print("\n🔢 Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_handler.generate_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # 6. Initialize vector store
        print("\n💾 Initializing vector store...")
        vector_store = VectorStore(
            embedding_dimension=embedding_handler.get_embedding_dimension(),
            index_path=str(VECTOR_DB_DIR)
        )
        
        # 7. Add documents to vector store
        print("\n📥 Adding documents to vector store...")
        vector_store.add_documents(embeddings, texts, chunks)
        
        # 8. Save index
        print("\n💾 Saving vector index...")
        vector_store.save()
        
        print("\n" + "="*60)
        print("✅ Vector Store Rebuilt Successfully!")
        print("="*60)
        print(f"\n📊 Final Stats:")
        stats = vector_store.get_stats()
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
        print(f"   Index saved to: {stats['index_path']}")
        print(f"\n🎉 You can now restart your Flask app!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error rebuilding vector store: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = rebuild_vector_store()
    sys.exit(0 if success else 1)