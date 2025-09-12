# rag/__init__.py
"""
RAG (Retrieval-Augmented Generation) Module
"""

from rag.embeddings import EmbeddingService
from rag.hybrid_search import HybridSearch

__all__ = [
    'EmbeddingService',
    'HybridSearch'
]

__version__ = '1.0.0'