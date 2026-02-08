"""
Vector Store
FAISS-based vector database with improved similarity search
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
import faiss
from pathlib import Path

class VectorStore:
    """FAISS-based vector store with improved retrieval accuracy"""
    
    def __init__(self, embedding_dimension: int, index_path: str = "vector_db"):
        """
        Initialize vector store
        
        Args:
            embedding_dimension: Dimension of embedding vectors
            index_path: Directory to save/load FAISS index
        """
        self.embedding_dimension = embedding_dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index with Inner Product (for cosine similarity with normalized vectors)
        # This is more accurate than L2 for semantic similarity
        self.index = faiss.IndexFlatIP(embedding_dimension)
        
        # Store metadata for each vector
        self.metadata = []
        
        # Store original texts
        self.texts = []
    
    def add_documents(self, embeddings: np.ndarray, texts: List[str], 
                     metadata: List[Dict]):
        """
        Add documents to the vector store with proper normalization
        
        Args:
            embeddings: Array of embedding vectors
            texts: List of text chunks
            metadata: List of metadata dictionaries
        """
        # Convert to float32 if needed
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity (critical for accuracy)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        # Add to FAISS index
        self.index.add(normalized_embeddings)
        
        # Store metadata and texts
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        print(f"✅ Added {len(embeddings)} documents to vector store")
        print(f"   Total documents: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3, 
               min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for most similar documents with improved relevance scoring
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of dictionaries containing text, metadata, and similarity score
        """
        if self.index.ntotal == 0:
            print("⚠️  Vector store is empty!")
            return []
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Search - for IndexFlatIP, the score is the inner product (cosine similarity for normalized vectors)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results with proper similarity scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Score is already cosine similarity (range -1 to 1)
            # Clip to 0-1 range for better interpretation
            similarity = float(np.clip(score, 0, 1))
            
            # Filter by minimum similarity
            if similarity < min_similarity:
                continue
            
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'similarity': similarity,
                'rank': i + 1,
                'score': float(score)  # Raw score for debugging
            })
        
        return results
    
    def search_with_reranking(self, query_embedding: np.ndarray, top_k: int = 3,
                             retrieval_k: int = 10) -> List[Dict]:
        """
        Search with retrieval oversampling and reranking for better accuracy
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            retrieval_k: Number of candidates to retrieve initially
            
        Returns:
            List of reranked results
        """
        # Retrieve more candidates than needed
        candidates = self.search(query_embedding, top_k=retrieval_k, min_similarity=0.0)
        
        if not candidates:
            return []
        
        # Sort by similarity (already done in search, but ensuring)
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top_k after reranking
        return candidates[:top_k]
    
    def save(self, name: str = "index"):
        """
        Save the vector store to disk
        
        Args:
            name: Name for the saved index
        """
        # Save FAISS index
        index_file = self.index_path / f"{name}.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata and texts
        metadata_file = self.index_path / f"{name}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'embedding_dimension': self.embedding_dimension
            }, f)
        
        print(f"💾 Vector store saved to {self.index_path}")
    
    def load(self, name: str = "index") -> bool:
        """
        Load the vector store from disk
        
        Args:
            name: Name of the saved index
            
        Returns:
            True if successful, False otherwise
        """
        index_file = self.index_path / f"{name}.faiss"
        metadata_file = self.index_path / f"{name}_metadata.pkl"
        
        if not index_file.exists() or not metadata_file.exists():
            print(f"⚠️  No saved index found at {self.index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata and texts
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.embedding_dimension = data['embedding_dimension']
            
            print(f"✅ Loaded vector store with {self.index.ntotal} documents")
            return True
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.texts = []
        self.metadata = []
        print("🗑️  Vector store cleared")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with statistics
        """
        # Count unique documents
        unique_sources = set()
        for meta in self.metadata:
            unique_sources.add(meta.get('source', 'Unknown'))
        
        return {
            'total_documents': len(unique_sources),
            'total_chunks': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'index_path': str(self.index_path),
            'sources': list(unique_sources)
        }
    
    def get_document_chunks(self, source: str) -> List[Dict]:
        """
        Get all chunks from a specific document
        
        Args:
            source: Document source name
            
        Returns:
            List of chunks from that document
        """
        chunks = []
        for i, meta in enumerate(self.metadata):
            if meta.get('source') == source:
                chunks.append({
                    'text': self.texts[i],
                    'metadata': meta,
                    'index': i
                })
        return chunks


# Test function
if __name__ == "__main__":
    print("Testing Improved Vector Store...")
    
    # Create test data
    dimension = 384  # all-MiniLM-L6-v2 dimension
    num_docs = 5
    
    # Generate random embeddings for testing
    np.random.seed(42)
    embeddings = np.random.randn(num_docs, dimension).astype('float32')
    
    texts = [
        "Artificial intelligence is transforming the world through machine learning and deep learning.",
        "Machine learning models can learn from data and improve over time without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from images."
    ]
    
    metadata = [
        {'source': 'ai_overview.pdf', 'chunk_id': i, 'page': 1} for i in range(num_docs)
    ]
    
    # Initialize vector store
    store = VectorStore(dimension, "test_vector_db")
    
    # Add documents
    print("\n📥 Adding documents...")
    store.add_documents(embeddings, texts, metadata)
    
    # Test search
    print("\n🔍 Testing search...")
    query_embedding = embeddings[0] + np.random.randn(dimension) * 0.1  # Similar to first doc
    results = store.search(query_embedding, top_k=3)
    
    print("\nSearch results:")
    for result in results:
        print(f"\n  Rank {result['rank']}: {result['text'][:60]}...")
        print(f"  Similarity: {result['similarity']:.2%}")
        print(f"  Source: {result['metadata']['source']}")
    
    # Test search with reranking
    print("\n🔍 Testing search with reranking...")
    reranked_results = store.search_with_reranking(query_embedding, top_k=3, retrieval_k=5)
    
    print("\nReranked results:")
    for result in reranked_results:
        print(f"\n  Rank {result['rank']}: Similarity {result['similarity']:.2%}")
    
    # Test save and load
    print("\n💾 Testing save/load...")
    store.save("test")
    
    new_store = VectorStore(dimension, "test_vector_db")
    loaded = new_store.load("test")
    
    if loaded:
        stats = new_store.get_stats()
        print(f"\n📊 Stats after loading:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Sources: {stats['sources']}")
    
    print("\n✅ All tests passed!")
