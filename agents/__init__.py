# agents/__init__.py
"""
Dental AI System - Multi-Agent Module
"""

from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.scheduler import SchedulerAgent
from agents.safety import SafetyAgent
from agents.summarizer import SummarizerAgent
from agents.orchestrator import MultiAgentOrchestrator

__all__ = [
    'BaseAgent',
    'PlannerAgent',
    'RetrieverAgent',
    'SchedulerAgent',
    'SafetyAgent',
    'SummarizerAgent',
    'MultiAgentOrchestrator'
]

__version__ = '1.0.0'