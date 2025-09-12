# agents/orchestrator.py
"""
Multi-agent orchestrator for complex task coordination
"""

from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
import asyncio

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.scheduler import SchedulerAgent
from agents.safety import SafetyAgent
from agents.summarizer import SummarizerAgent

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Orchestrates multiple agents to handle complex tasks"""
    
    def __init__(
        self,
        openai_api_key: str,
        embedding_service=None,
        hybrid_search=None
    ):
        """
        Initialize orchestrator with all agents
        
        Args:
            openai_api_key: OpenAI API key
            embedding_service: Embedding service instance
            hybrid_search: Hybrid search instance
        """
        
        # Initialize agents
        self.agents = {
            'planner': PlannerAgent(openai_api_key),
            'retriever': RetrieverAgent(openai_api_key, hybrid_search),
            'scheduler': SchedulerAgent(openai_api_key),
            'safety': SafetyAgent(),
            'summarizer': SummarizerAgent(openai_api_key)
        }
        
        self.execution_trace = []
        self.total_tokens = 0
        self.total_cost = 0.0
    
    async def execute(
        self,
        task: str,
        tenant_id: str,
        user_id: str,
        user_role: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complex task using multiple agents
        
        Args:
            task: Task description
            tenant_id: Tenant ID
            user_id: User ID
            user_role: User role (patient, staff, admin)
            parameters: Additional parameters
            
        Returns:
            Execution result with trace
        """
        
        start_time = datetime.utcnow()
        self.execution_trace = []
        self.total_tokens = 0
        self.total_cost = 0.0
        
        try:
            # Step 1: Safety check
            safety_agent = self.agents['safety']
            validation = safety_agent.validate_request(task, user_role)
            
            if not validation['valid']:
                return {
                    'final_result': {
                        'error': validation['reason'],
                        'status': 'blocked'
                    },
                    'trace': [validation],
                    'total_tokens': 0,
                    'total_cost': 0.0
                }
            
            # Step 2: Plan the task
            planner = self.agents['planner']
            context = {
                'tenant_id': tenant_id,
                'user_id': user_id,
                'user_role': user_role,
                'parameters': parameters or {}
            }
            
            plan = await planner.execute(task, context)
            self._log_step('planner', plan)
            
            # Step 3: Execute plan steps
            step_results = {}
            
            for step in plan.get('steps', []):
                agent_name = step['agent']
                agent_task = step['task']
                
                # Get agent
                agent = self.agents.get(agent_name)
                if not agent:
                    logger.warning(f"Agent {agent_name} not found")
                    continue
                
                # Prepare context with previous results
                step_context = {
                    **context,
                    'step': step,
                    'previous_results': step_results
                }
                
                # Execute agent task
                if agent_name == 'safety':
                    # Safety agent has different interface
                    result = safety_agent.sanitize_output(
                        agent_task,
                        user_role,
                        tenant_id
                    )
                else:
                    result = await agent.execute(agent_task, step_context)
                
                step_results[step['id']] = result
                self._log_step(agent_name, result)
            
            # Step 4: Summarize results
            summarizer = self.agents['summarizer']
            final_summary = await summarizer.execute(
                "Summarize the results",
                {
                    'task': task,
                    'results': step_results,
                    'tenant_id': tenant_id
                }
            )
            
            self._log_step('summarizer', final_summary)
            
            # Calculate final metrics
            elapsed_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'final_result': final_summary,
                'trace': self.execution_trace,
                'total_tokens': self.total_tokens,
                'total_cost': self._calculate_cost(),
                'execution_time_ms': elapsed_time * 1000
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return {
                'final_result': {
                    'error': str(e),
                    'status': 'error'
                },
                'trace': self.execution_trace,
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost
            }
    
    def _log_step(self, agent_name: str, result: Dict[str, Any]):
        """
        Log execution step
        
        Args:
            agent_name: Name of agent
            result: Execution result
        """
        
        trace_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent': agent_name,
            'result': result.get('answer', result) if isinstance(result, dict) else result,
            'tokens': result.get('tokens_used', 0) if isinstance(result, dict) else 0
        }
        
        self.execution_trace.append(trace_entry)
        
        # Update token count
        if isinstance(result, dict) and 'tokens_used' in result:
            self.total_tokens += result['tokens_used']
    
    def _calculate_cost(self) -> float:
        """
        Calculate total cost based on token usage
        
        Returns:
            Estimated cost in USD
        """
        
        # Pricing per 1K tokens (approximate)
        pricing = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4o': {'input': 0.005, 'output': 0.015}
        }
        
        # Simple estimation (assuming 50/50 input/output split)
        base_price = pricing['gpt-4o-mini']
        cost_per_token = (base_price['input'] + base_price['output']) / 2000
        
        return self.total_tokens * cost_per_token