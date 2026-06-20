"""
RAG Engine for Content Agent
Implements Retrieval-Augmented Generation using ChromaDB and embeddings.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for fact-based content creation.

    Features:
    - Document ingestion and chunking
    - Semantic search with vector embeddings
    - Multi-source knowledge base
    - Citation tracking
    """

    def __init__(
        self,
        collection_name: str = "content_knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./data/chroma_db"
    ):
        """
        Initialize RAG engine with vector database.

        Args:
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model for embeddings
            persist_directory: Path to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Knowledge base for content generation"}
            )
            logger.info(f"Created new collection: {collection_name}")

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    async def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text content
            metadata: Optional metadata (source, author, date, etc.)
            doc_id: Optional custom document ID

        Returns:
            Document ID in the collection
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Document text is too short or empty")

        # Generate document ID if not provided
        if not doc_id:
            import hashlib
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]

        # Split document into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")

        # Generate embeddings for each chunk
        embeddings = self.embedding_model.encode(chunks).tolist()

        # Prepare metadata for each chunk
        chunk_metadata = metadata or {}
        chunk_metadata["doc_id"] = doc_id
        chunk_metadata["total_chunks"] = len(chunks)

        # Add chunks to collection
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{**chunk_metadata, "chunk_index": i} for i in range(len(chunks))]
        )

        logger.success(f"Added document {doc_id} with {len(chunks)} chunks")
        return doc_id

    async def add_documents_from_directory(
        self,
        directory_path: str,
        file_extensions: List[str] = [".txt", ".md"]
    ) -> List[str]:
        """
        Batch add documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to process

        Returns:
            List of added document IDs
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        added_docs = []

        for ext in file_extensions:
            for file_path in directory.glob(f"**/*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    metadata = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "extension": ext
                    }

                    doc_id = await self.add_document(text, metadata)
                    added_docs.append(doc_id)

                    logger.info(f"Added: {file_path.name}")

                except Exception as e:
                    logger.error(f"Failed to add {file_path}: {e}")

        logger.success(f"Added {len(added_docs)} documents from {directory_path}")
        return added_docs

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant document chunks with metadata
        """
        if not query or len(query.strip()) < 3:
            raise ValueError("Query is too short")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        retrieved_docs = []

        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "source": results['metadatas'][0][i].get('source', 'Unknown') if results['metadatas'] else 'Unknown'
                })

        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
        return retrieved_docs

    async def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword-based retrieval.

        Args:
            query: Semantic search query
            keywords: Keyword filters
            top_k: Number of results

        Returns:
            Combined search results
        """
        # Semantic search
        semantic_results = await self.retrieve(query, top_k=top_k)

        # Keyword filtering (simple implementation)
        # In production, use BM25 or Elasticsearch for better keyword search
        filtered_results = []

        for doc in semantic_results:
            text_lower = doc['text'].lower()
            keyword_match_score = sum(
                1 for kw in keywords if kw.lower() in text_lower
            )

            if keyword_match_score > 0:
                doc['keyword_matches'] = keyword_match_score
                filtered_results.append(doc)

        # Sort by keyword matches (descending)
        filtered_results.sort(
            key=lambda x: x.get('keyword_matches', 0),
            reverse=True
        )

        logger.info(f"Hybrid search returned {len(filtered_results)} results")
        return filtered_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "persist_directory": self.persist_directory
        }

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the knowledge base.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted document {doc_id} with {len(results['ids'])} chunks")
                return True
            else:
                logger.warning(f"Document {doc_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base for content generation"}
            )
            logger.warning(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo RAG engine"""
        rag = RAGEngine()

        # Add sample documents
        print("\n" + "="*80)
        print("Adding documents to knowledge base...")
        print("="*80)

        sample_docs = [
            {
                "text": """AI agents are autonomous software systems that can perceive their
                environment, make decisions, and take actions to achieve specific goals.
                They use machine learning and natural language processing to understand
                and respond to complex tasks.""",
                "metadata": {"source": "AI Handbook", "topic": "AI Agents"}
            },
            {
                "text": """Content marketing in 2024 requires a data-driven approach.
                Successful content strategies focus on SEO optimization, audience targeting,
                and multi-channel distribution. AI tools are increasingly used to automate
                content creation and optimization.""",
                "metadata": {"source": "Marketing Guide", "topic": "Content Marketing"}
            },
            {
                "text": """RAG (Retrieval-Augmented Generation) combines information retrieval
                with text generation. It allows language models to access external knowledge
                bases, reducing hallucinations and improving factual accuracy.""",
                "metadata": {"source": "ML Research", "topic": "RAG"}
            }
        ]

        for doc in sample_docs:
            await rag.add_document(doc["text"], doc["metadata"])

        # Retrieve relevant information
        print("\n" + "="*80)
        print("Searching knowledge base...")
        print("="*80)

        query = "How do AI agents work?"
        results = await rag.retrieve(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Text: {result['text'][:200]}...")
            print(f"Source: {result['source']}")
            print(f"Distance: {result['distance']:.4f}")

        # Get stats
        print("\n" + "="*80)
        print("Knowledge Base Stats:")
        print("="*80)
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    asyncio.run(demo())
