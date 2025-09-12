# rag/embeddings.py
"""
Embedding service for document vectorization
"""

import openai
from typing import List, Dict, Any
import numpy as np
import tiktoken
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using OpenAI"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize embedding service
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }.get(model, 1536)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Truncate if too long
            text = self._truncate_text(text)
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Truncate all texts
            texts = [self._truncate_text(text) for text in texts]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.encoding.decode(tokens)
        return text
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "tokens": len(self.encoding.encode(chunk_text))
                })
                chunk_index += 1
            
            start = end - overlap if end < text_length else end
        
        return chunks