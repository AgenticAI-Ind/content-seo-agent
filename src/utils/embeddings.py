"""
Vector Embedding Utilities
Helper functions for text embeddings and similarity.
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class EmbeddingUtil:
    """
    Utility class for text embeddings and similarity calculations.

    Features:
    - Text to vector conversion
    - Similarity search
    - Batch processing
    - Model caching
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding utility.

        Args:
            model_name: Sentence transformer model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.success(f"Model loaded: {self.embedding_dim}D embeddings")

    def encode(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert multiple texts to embeddings.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Array of embedding vectors
        """
        logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")

        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        logger.info(f"Finding top {top_k} similar texts from {len(candidates)} candidates")

        # Encode query and candidates
        query_embedding = self.encode(query)
        candidate_embeddings = self.encode_batch(candidates)

        # Calculate similarities
        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(query_embedding, candidate_emb)
            similarities.append((candidates[i], similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def semantic_search(
        self,
        query: str,
        documents: List[str],
        threshold: float = 0.5
    ) -> List[Tuple[int, str, float]]:
        """
        Semantic search over documents.

        Args:
            query: Search query
            documents: List of documents to search
            threshold: Minimum similarity threshold

        Returns:
            List of (index, document, score) tuples
        """
        logger.info(f"Semantic search over {len(documents)} documents")

        query_embedding = self.encode(query)
        doc_embeddings = self.encode_batch(documents)

        results = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_emb)

            if similarity >= threshold:
                results.append((i, documents[i], similarity))

        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def cluster_texts(
        self,
        texts: List[str],
        num_clusters: int = 5
    ) -> List[int]:
        """
        Cluster texts based on semantic similarity.

        Args:
            texts: List of texts to cluster
            num_clusters: Number of clusters

        Returns:
            List of cluster assignments
        """
        from sklearn.cluster import KMeans

        logger.info(f"Clustering {len(texts)} texts into {num_clusters} clusters")

        # Get embeddings
        embeddings = self.encode_batch(texts)

        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        return clusters.tolist()

    def deduplicate(
        self,
        texts: List[str],
        similarity_threshold: float = 0.95
    ) -> List[str]:
        """
        Remove duplicate or near-duplicate texts.

        Args:
            texts: List of texts
            similarity_threshold: Similarity threshold for duplicates

        Returns:
            List of unique texts
        """
        logger.info(f"Deduplicating {len(texts)} texts")

        if not texts:
            return []

        embeddings = self.encode_batch(texts)

        unique_texts = [texts[0]]
        unique_embeddings = [embeddings[0]]

        for i in range(1, len(texts)):
            is_duplicate = False

            for unique_emb in unique_embeddings:
                similarity = self.cosine_similarity(embeddings[i], unique_emb)

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_texts.append(texts[i])
                unique_embeddings.append(embeddings[i])

        logger.info(f"Reduced from {len(texts)} to {len(unique_texts)} unique texts")

        return unique_texts


# Example usage
if __name__ == "__main__":
    util = EmbeddingUtil()

    # Sample texts
    query = "AI agents for content creation"
    candidates = [
        "Using artificial intelligence for writing blog posts",
        "Machine learning models for text generation",
        "How to cook pasta at home",
        "Automated content generation with AI",
        "Best restaurants in New York"
    ]

    # Find similar texts
    results = util.find_most_similar(query, candidates, top_k=3)

    print(f"Query: {query}\n")
    print("Most similar texts:")
    for text, score in results:
        print(f"  {score:.3f} - {text}")
