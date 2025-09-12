# agents/retriever.py
"""
Retriever agent for RAG-based information retrieval
"""

from typing import Dict, Any, List
import logging
from agents.base import BaseAgent

logger = logging.getLogger(__name__)

class RetrieverAgent(BaseAgent):
    """Agent specialized in retrieving and synthesizing information"""
    
    def __init__(self, openai_api_key: str, hybrid_search=None):
        """
        Initialize retriever agent
        
        Args:
            openai_api_key: OpenAI API key
            hybrid_search: Hybrid search instance
        """
        super().__init__(
            name="Retriever",
            description="Information retrieval specialist that searches documents and provides grounded answers with citations",
            openai_api_key=openai_api_key
        )
        self.hybrid_search = hybrid_search
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute retrieval task
        
        Args:
            task: Retrieval query
            context: Context including tenant_id, filters, etc.
            
        Returns:
            Retrieved information with citations
        """
        try:
            # Extract parameters from context
            tenant_id = context.get("tenant_id")
            filters = context.get("filters", {})
            top_k = context.get("top_k", 5)
            
            # Perform search if hybrid_search is available
            search_results = []
            if self.hybrid_search and tenant_id:
                search_results = await self.hybrid_search.search(
                    query=task,
                    tenant_id=tenant_id,
                    filters=filters,
                    top_k=top_k
                )
            
            # Generate answer with citations
            result = await self.answer_with_citations(
                query=task,
                search_results=search_results,
                tenant_id=tenant_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Retriever agent error: {str(e)}")
            return {
                "answer": "I encountered an error while searching for information.",
                "citations": [],
                "error": str(e)
            }
    
    async def answer_with_citations(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Generate answer with proper citations
        
        Args:
            query: User query
            search_results: Search results from hybrid search
            tenant_id: Tenant ID
            
        Returns:
            Answer with citations
        """
        
        # If no results, return appropriate message
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in our documentation to answer your question.",
                "citations": [],
                "confidence": 0.0,
                "tokens_used": 0
            }
        
        # Format search results for prompt
        context_docs = self._format_search_results(search_results)
        
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self._build_retrieval_prompt()},
            {"role": "user", "content": f"""
Based on the following documents from our dental practice database, please answer the question.

Documents:
{context_docs}

Question: {query}

Provide a clear, accurate answer using ONLY the information from the documents above. 
Include [1], [2], etc. to cite specific documents.
If the answer is not in the documents, say so clearly."""}
        ]
        
        # Call LLM
        response = await self._call_llm(messages, temperature=0.3, max_tokens=1000)
        
        # Calculate confidence based on result quality
        confidence = self._calculate_confidence(search_results)
        
        return {
            "answer": response["content"],
            "citations": self._extract_citations(search_results[:3]),  # Top 3 citations
            "confidence": confidence,
            "tokens_used": response["usage"]["total_tokens"],
            "search_results_count": len(search_results)
        }
    
    def _build_retrieval_prompt(self) -> str:
        """Build specialized prompt for retrieval"""
        return """You are a knowledgeable dental practice assistant. Your role is to provide accurate information based solely on the provided documents.

Important rules:
1. Only use information explicitly stated in the provided documents
2. Use citations [1], [2], etc. to reference specific documents
3. If information is not in the documents, clearly state that
4. Be concise but thorough
5. Maintain a professional, helpful tone
6. Never make up or infer information not in the documents"""
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for prompt
        
        Args:
            results: Search results
            
        Returns:
            Formatted string
        """
        formatted = []
        for i, result in enumerate(results[:5], 1):  # Use top 5 results
            formatted.append(f"""[{i}] {result.get('doc_title', 'Document')} ({result.get('doc_type', 'general')})
Content: {result.get('content', '')}
---""")
        
        return "\n".join(formatted)
    
    def _extract_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citations from search results
        
        Args:
            results: Search results
            
        Returns:
            List of citations
        """
        citations = []
        for result in results:
            citations.append({
                "doc_id": result.get("doc_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "title": result.get("doc_title", ""),
                "content": result.get("content", "")[:200] + "...",
                "relevance_score": result.get("final_score", result.get("score", 0.0))
            })
        return citations
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on search results
        
        Args:
            results: Search results
            
        Returns:
            Confidence score between 0 and 1
        """
        if not results:
            return 0.0
        
        # Get top score
        top_score = results[0].get("final_score", results[0].get("score", 0.0))
        
        # Calculate based on top score and number of relevant results
        relevant_count = sum(1 for r in results if r.get("score", 0) > 0.5)
        
        confidence = min(
            (top_score * 0.7) + (min(relevant_count / 3, 1.0) * 0.3),
            1.0
        )
        
        return confidence