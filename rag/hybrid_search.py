# rag/hybrid_search.py
"""
Simplified hybrid search implementation for free tier
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HybridSearch:
    """Simplified hybrid search using in-memory storage"""
    
    def __init__(self, embedding_service, database_url: str = None):
        """
        Initialize hybrid search
        
        Args:
            embedding_service: Service for generating embeddings
            database_url: PostgreSQL connection string (optional for now)
        """
        self.embedding_service = embedding_service
        self.database_url = database_url
        
        # Use in-memory storage for simplicity
        self.documents = []
        self.document_embeddings = []
        
        # Load mock documents
        self._load_mock_documents()
    
    async def initialize(self):
        """Initialize storage"""
        logger.info("Hybrid search initialized with in-memory storage")
        return True
    
    async def close(self):
        """Close connections"""
        pass
    
    def _load_mock_documents(self):
        """Load mock documents for demo"""
        mock_docs = [
            {
                "chunk_id": "chunk_001",
                "doc_id": "doc_001",
                "content": "Our office hours are Monday through Friday from 8:00 AM to 5:00 PM, and Saturday from 9:00 AM to 2:00 PM. We are closed on Sundays.",
                "doc_title": "Office Hours",
                "doc_type": "policy",
                "tenant_id": "11111111-1111-1111-1111-111111111111"
            },
            {
                "chunk_id": "chunk_002",
                "doc_id": "doc_002",
                "content": "We accept most major dental insurance plans including Delta Dental, Aetna, and Cigna. Preventive care is typically covered at 100%.",
                "doc_title": "Insurance Coverage",
                "doc_type": "policy",
                "tenant_id": "11111111-1111-1111-1111-111111111111"
            },
            {
                "chunk_id": "chunk_003",
                "doc_id": "doc_003",
                "content": "Root canal treatment is a procedure to repair and save a badly damaged or infected tooth. Most insurance plans cover 50-80% of the cost.",
                "doc_title": "Root Canal Treatment",
                "doc_type": "procedure",
                "tenant_id": "11111111-1111-1111-1111-111111111111"
            },
            {
                "chunk_id": "chunk_004",
                "doc_id": "doc_004",
                "content": "For dental emergencies, we offer same-day appointments when available. Please call our emergency line at (555) 123-4567.",
                "doc_title": "Emergency Care",
                "doc_type": "faq",
                "tenant_id": "11111111-1111-1111-1111-111111111111"
            },
            {
                "chunk_id": "chunk_005",
                "doc_id": "doc_005",
                "content": "Professional teeth cleaning is recommended every six months. The appointment typically takes 30-60 minutes.",
                "doc_title": "Teeth Cleaning",
                "doc_type": "procedure",
                "tenant_id": "11111111-1111-1111-1111-111111111111"
            }
        ]
        
        self.documents = mock_docs
        # We'll generate embeddings on first search
    
    async def search(
        self,
        query: str,
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform simplified search
        
        Args:
            query: Search query
            tenant_id: Tenant ID for filtering
            filters: Additional filters
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Filter documents by tenant
            tenant_docs = [
                doc for doc in self.documents 
                if doc.get("tenant_id") == tenant_id
            ]
            
            if not tenant_docs:
                return []
            
            # Apply additional filters
            if filters and filters.get("doc_types"):
                tenant_docs = [
                    doc for doc in tenant_docs 
                    if doc.get("doc_type") in filters["doc_types"]
                ]
            
            # Simple keyword matching for now (to save API calls)
            query_lower = query.lower()
            scored_docs = []
            
            for doc in tenant_docs:
                content_lower = doc["content"].lower()
                title_lower = doc.get("doc_title", "").lower()
                
                # Calculate simple relevance score
                score = 0
                query_words = query_lower.split()
                
                for word in query_words:
                    if word in content_lower:
                        score += content_lower.count(word) * 2
                    if word in title_lower:
                        score += 5
                
                if score > 0:
                    doc_copy = doc.copy()
                    doc_copy["score"] = score / (len(query_words) + 1)
                    doc_copy["final_score"] = doc_copy["score"]
                    doc_copy["search_type"] = "keyword"
                    doc_copy["metadata"] = {}
                    scored_docs.append(doc_copy)
            
            # Sort by score
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    async def add_document(
        self,
        tenant_id: str,
        doc_type: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the in-memory store
        
        Args:
            tenant_id: Tenant ID
            doc_type: Document type
            title: Document title
            content: Document content
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        
        import uuid
        
        doc_id = str(uuid.uuid4())
        chunk_id = f"chunk_{len(self.documents) + 1:03d}"
        
        self.documents.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "doc_title": title,
            "doc_type": doc_type,
            "tenant_id": tenant_id,
            "metadata": metadata or {}
        })
        
        logger.info(f"Added document {doc_id} to in-memory store")
        return doc_id