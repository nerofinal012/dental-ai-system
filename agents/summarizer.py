# agents/summarizer.py
"""
Summarizer agent for consolidating information
"""

from typing import Dict, Any, List
import logging
from agents.base import BaseAgent

logger = logging.getLogger(__name__)

class SummarizerAgent(BaseAgent):
    """Agent specialized in summarizing and consolidating information"""
    
    def __init__(self, openai_api_key: str):
        """Initialize summarizer agent"""
        super().__init__(
            name="Summarizer",
            description="Information synthesis specialist that creates clear, actionable summaries",
            openai_api_key=openai_api_key
        )
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute summarization task
        
        Args:
            task: Summarization request
            context: Context including results to summarize
            
        Returns:
            Summarized information
        """
        try:
            # Extract information to summarize
            results = context.get('results', {})
            original_task = context.get('task', task)
            
            # Build summarization prompt
            messages = [
                {"role": "system", "content": self._build_summarizer_prompt()},
                {"role": "user", "content": f"""
Original request: {original_task}

Information gathered:
{self._format_results(results)}

Please provide a clear, concise summary that directly answers the original request.
Include key points and any necessary action items."""}
            ]
            
            response = await self._call_llm(messages, temperature=0.3)
            
            return {
                "answer": response["content"],
                "tokens_used": response["usage"]["total_tokens"],
                "summary_type": "consolidated",
                "sections": self._extract_sections(response["content"])
            }
            
        except Exception as e:
            logger.error(f"Summarizer error: {str(e)}")
            return {
                "answer": "Unable to generate summary.",
                "error": str(e)
            }
    
    def _build_summarizer_prompt(self) -> str:
        """Build summarizer system prompt"""
        return """You are a medical communication specialist for a dental practice.

Your role is to create clear, actionable summaries that:
1. Directly answer the patient's question
2. Highlight key information
3. Provide clear next steps when applicable
4. Maintain a professional yet friendly tone

Structure your summaries with:
- A direct answer (1-2 sentences)
- Key points (if multiple items)
- Action items (if any)
- Contact information (when relevant)

Guidelines:
- Use simple language, avoid medical jargon
- Be concise but complete
- Emphasize important dates, times, or requirements
- Always maintain HIPAA compliance"""
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Format results for summarization
        
        Args:
            results: Results from various agents
            
        Returns:
            Formatted string
        """
        if not results:
            return "No previous results available."
        
        formatted = []
        
        for step_id, result in results.items():
            if isinstance(result, dict):
                if 'answer' in result:
                    formatted.append(f"[{step_id}]: {result['answer']}")
                elif 'content' in result:
                    formatted.append(f"[{step_id}]: {result['content']}")
                else:
                    formatted.append(f"[{step_id}]: {str(result)[:200]}")
            else:
                formatted.append(f"[{step_id}]: {str(result)[:200]}")
        
        return "\n\n".join(formatted)
    
    def _extract_sections(self, summary: str) -> Dict[str, str]:
        """
        Extract sections from summary
        
        Args:
            summary: Summary text
            
        Returns:
            Dictionary of sections
        """
        sections = {
            "main": summary,
            "has_action_items": "call" in summary.lower() or "book" in summary.lower(),
            "has_dates": any(char.isdigit() for char in summary)
        }
        
        return sections