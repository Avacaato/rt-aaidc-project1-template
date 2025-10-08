import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
        )

        chunks = splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """

        print(f"Processing {len(documents)} documents...")
        for doc_id, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(content)

            print(f"Document {doc_id}: Split into {len(chunks)} chunks")

            # Create unique IDs for each chunk
            ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]

            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(chunks)

            # Add to ChromaDB collection
            self.collection.add(
                documents=chunks,
                metadatas=[metadata] * len(chunks),
                embeddings=embeddings.tolist(),
                ids=ids,
            )
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances", "ids"],
        )

        if results and all(key in results for key in ["documents", "metadatas", "distances", "ids"]):
            return results
        else:
        
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }
