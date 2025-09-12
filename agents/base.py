# agents/base.py
"""
Base agent class for all specialized agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import openai
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(
        self,
        name: str,
        description: str,
        openai_api_key: str,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            description: Agent description and role
            openai_api_key: OpenAI API key
            model: LLM model to use
        """
        self.name = name
        self.description = description
        self.model = model
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.execution_history = []
        
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent task
        
        Args:
            task: Task description
            context: Execution context
            
        Returns:
            Execution result
        """
        pass
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call OpenAI LLM
        
        Args:
            messages: Chat messages
            temperature: Temperature for sampling
            max_tokens: Maximum tokens in response
            response_format: Expected response format (e.g., "json")
            
        Returns:
            LLM response with metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add JSON mode if requested
            if response_format == "json":
                request_params["response_format"] = {"type": "json_object"}
            
            # Make API call
            response = self.client.chat.completions.create(**request_params)
            
            # Calculate metrics
            elapsed_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract response
            content = response.choices[0].message.content
            
            # Parse JSON if expected
            if response_format == "json":
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response from {self.name}")
            
            # Log execution
            self._log_execution(task="llm_call", elapsed_time=elapsed_time, tokens=response.usage.total_tokens)
            
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "latency_ms": elapsed_time * 1000
            }
            
        except Exception as e:
            logger.error(f"LLM call failed in {self.name}: {str(e)}")
            raise
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for the agent
        
        Returns:
            System prompt
        """
        return f"""You are {self.name}, a specialized agent in a dental practice management system.
        
Role: {self.description}

Guidelines:
- Be precise and accurate in your responses
- Always base answers on provided information
- Maintain patient privacy and HIPAA compliance
- Provide clear, actionable responses
- Format responses appropriately for the task"""
    
    def _log_execution(self, task: str, elapsed_time: float, tokens: int = 0):
        """
        Log execution details
        
        Args:
            task: Task executed
            elapsed_time: Time taken in seconds
            tokens: Tokens used
        """
        log_entry = {
            "agent": self.name,
            "task": task,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": elapsed_time * 1000,
            "tokens": tokens
        }
        
        self.execution_history.append(log_entry)
        logger.info(f"Agent {self.name} executed {task} in {elapsed_time:.2f}s")
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """
        Get execution trace for this agent
        
        Returns:
            List of execution entries
        """
        return self.execution_history.copy()
    
    def reset_trace(self):
        """Reset execution history"""
        self.execution_history = []