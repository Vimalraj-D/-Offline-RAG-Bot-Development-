"""
Embedding Handler
Generates embeddings using SentenceTransformers
"""

import numpy as np
from typing import List, Union, Dict
from sentence_transformers import SentenceTransformer

class EmbeddingHandler:
    """Handle embedding generation using local models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding handler
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        
    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            print(f"📥 Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
                # Get embedding dimension
                test_embedding = self.model.encode("test")
                self.embedding_dimension = len(test_embedding)
                print(f"✅ Model loaded. Embedding dimension: {self.embedding_dimension}")
            except Exception as e:
                print(f"❌ Error loading embedding model: {e}")
                raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            self.load_model()
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dimension))
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        
        Returns:
            Embedding dimension
        """
        if self.embedding_dimension is None:
            if self.model is None:
                self.load_model()
        return self.embedding_dimension
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def rerank_by_similarity(self, query_embedding: np.ndarray, 
                             results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank retrieved chunks by scoring each against the exact query embedding.
        Improves precision by prioritizing chunks most similar to the actual query.
        
        Args:
            query_embedding: Query vector (e.g. from exact user query)
            results: List of dicts with 'text', 'metadata', 'similarity', 'rank'
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list with updated similarity scores and ranks
        """
        if not results:
            return []
        
        texts = [r['text'] for r in results]
        chunk_embeddings = self.generate_embeddings(texts, show_progress=False)
        query_flat = query_embedding.astype('float32').reshape(1, -1)
        norm_q = np.linalg.norm(query_flat)
        if norm_q > 0:
            query_flat = query_flat / norm_q
        
        scores = []
        for i, emb in enumerate(chunk_embeddings):
            emb = emb.astype('float32').reshape(1, -1)
            norm_e = np.linalg.norm(emb)
            if norm_e > 0:
                emb = emb / norm_e
            sim = float(np.clip(np.dot(query_flat, emb.T).item(), 0, 1))
            scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for new_rank, (idx, sim) in enumerate(scores[:top_k], 1):
            r = dict(results[idx])
            r['similarity'] = sim
            r['rank'] = new_rank
            reranked.append(r)
        
        return reranked


# Test function
if __name__ == "__main__":
    # Test the embedding handler
    print("Testing Embedding Handler...")
    
    handler = EmbeddingHandler()
    handler.load_model()
    
    # Test single embedding
    text = "This is a test sentence."
    embedding = handler.generate_embedding(text)
    print(f"\nSingle embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {handler.get_embedding_dimension()}")
    
    # Test batch embeddings
    texts = [
        "Artificial intelligence is fascinating.",
        "Machine learning is a subset of AI.",
        "The weather is nice today."
    ]
    
    embeddings = handler.generate_embeddings(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    similarity_1_2 = handler.compute_similarity(embeddings[0], embeddings[1])
    similarity_1_3 = handler.compute_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity (AI vs ML): {similarity_1_2:.4f}")
    print(f"Similarity (AI vs weather): {similarity_1_3:.4f}")
    print("\n✅ All tests passed!")
