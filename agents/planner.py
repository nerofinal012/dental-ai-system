# agents/planner.py
"""
Planner agent for task decomposition and routing
"""

from typing import Dict, Any, List
import json
import logging
from agents.base import BaseAgent

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """Agent that decomposes complex queries into subtasks"""
    
    def __init__(self, openai_api_key: str):
        """Initialize planner agent"""
        super().__init__(
            name="Planner",
            description="Task decomposition specialist that breaks down complex queries into actionable steps",
            openai_api_key=openai_api_key,
            model="gpt-4o"  # Use advanced model for planning
        )
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create execution plan for task
        
        Args:
            task: Task description
            context: Execution context
            
        Returns:
            Execution plan with steps
        """
        try:
            # Build planning prompt
            messages = [
                {"role": "system", "content": self._build_planning_prompt()},
                {"role": "user", "content": f"""
Create an execution plan for this task: {task}

Context:
- User Role: {context.get('user_role', 'patient')}
- Available Agents: retriever (search docs), scheduler (appointments), safety (PHI check), summarizer (consolidate)

Output a JSON plan with steps."""}
            ]
            
            # Get plan from LLM
            response = await self._call_llm(
                messages,
                temperature=0.5,
                response_format="json"
            )
            
            # Parse and validate plan
            plan = response["content"]
            
            # Ensure plan has required structure
            if not isinstance(plan, dict):
                plan = {"steps": [], "error": "Invalid plan format"}
            
            if "steps" not in plan:
                # Create simple default plan
                plan = self._create_default_plan(task, context)
            
            return plan
            
        except Exception as e:
            logger.error(f"Planner error: {str(e)}")
            return self._create_default_plan(task, context)
    
    def _build_planning_prompt(self) -> str:
        """Build planning system prompt"""
        return """You are a task planner for a dental practice management system.

Your job is to decompose complex queries into simple steps that can be executed by specialist agents.

Available agents:
1. retriever: Searches documents and provides information with citations
2. scheduler: Manages appointments and availability
3. safety: Checks for PHI and ensures compliance
4. summarizer: Consolidates information from multiple sources

Output format (JSON):
{
  "analysis": "Brief analysis of the task",
  "steps": [
    {
      "id": "step_1",
      "agent": "agent_name",
      "task": "specific task description",
      "dependencies": []
    }
  ],
  "complexity": "low|medium|high"
}

Rules:
- Keep plans simple (2-4 steps typically)
- Always include safety check for sensitive data
- Use retriever for any information lookup
- Use scheduler for appointment-related tasks
- End with summarizer for multi-step tasks"""
    
    def _create_default_plan(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a default plan when planning fails
        
        Args:
            task: Original task
            context: Context
            
        Returns:
            Default plan
        """
        
        # Detect task type
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['appointment', 'schedule', 'book', 'available']):
            # Scheduling task
            return {
                "analysis": "Scheduling-related query",
                "steps": [
                    {
                        "id": "step_1",
                        "agent": "scheduler",
                        "task": task,
                        "dependencies": []
                    }
                ],
                "complexity": "low"
            }
        
        elif any(word in task_lower for word in ['insurance', 'coverage', 'claim', 'cost']):
            # Insurance/billing task
            return {
                "analysis": "Insurance or billing query",
                "steps": [
                    {
                        "id": "step_1",
                        "agent": "retriever",
                        "task": f"Find information about: {task}",
                        "dependencies": []
                    },
                    {
                        "id": "step_2",
                        "agent": "summarizer",
                        "task": "Summarize the findings",
                        "dependencies": ["step_1"]
                    }
                ],
                "complexity": "medium"
            }
        
        else:
            # General information query
            return {
                "analysis": "General information query",
                "steps": [
                    {
                        "id": "step_1",
                        "agent": "retriever",
                        "task": task,
                        "dependencies": []
                    }
                ],
                "complexity": "low"
            }