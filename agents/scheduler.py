# agents/scheduler.py
"""
Scheduler agent for appointment management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from agents.base import BaseAgent

logger = logging.getLogger(__name__)

class SchedulerAgent(BaseAgent):
    """Agent specialized in appointment scheduling"""
    
    def __init__(self, openai_api_key: str):
        """Initialize scheduler agent"""
        super().__init__(
            name="Scheduler",
            description="Appointment scheduling specialist that manages availability and bookings",
            openai_api_key=openai_api_key
        )
        
        # Mock available slots (in production, would query database)
        self.mock_slots = self._generate_mock_slots()
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scheduling task
        
        Args:
            task: Scheduling request
            context: Context including preferences, tenant_id, etc.
            
        Returns:
            Scheduling recommendations
        """
        try:
            # Get available slots
            tenant_id = context.get('tenant_id')
            preferences = context.get('parameters', {})
            
            available_slots = self._get_available_slots(
                tenant_id,
                preferences.get('preferred_dates'),
                preferences.get('preferred_times')
            )
            
            # Generate recommendations using LLM
            messages = [
                {"role": "system", "content": self._build_scheduler_prompt()},
                {"role": "user", "content": f"""
Task: {task}

Available appointment slots:
{self._format_slots(available_slots)}

Patient preferences:
- Preferred times: {preferences.get('preferred_times', 'Any time')}
- Urgency: {preferences.get('urgency', 'routine')}

Please recommend the best appointment options and explain why."""}
            ]
            
            response = await self._call_llm(messages, temperature=0.3)
            
            return {
                "answer": response["content"],
                "available_slots": available_slots[:5],  # Top 5 slots
                "tokens_used": response["usage"]["total_tokens"],
                "booking_instructions": "To book, please call (555) 123-4567 or use our online portal."
            }
            
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
            return {
                "answer": "I encountered an error while checking availability.",
                "available_slots": [],
                "error": str(e)
            }
    
    def _build_scheduler_prompt(self) -> str:
        """Build scheduler system prompt"""
        return """You are a dental appointment scheduling assistant.

Your responsibilities:
1. Recommend appropriate appointment slots based on availability
2. Consider patient preferences and urgency
3. Explain scheduling policies when relevant
4. Provide clear booking instructions

Guidelines:
- Emergency cases should be prioritized for same-day or next-day
- Routine cleanings can be scheduled further out
- Consider travel time between appointments
- Be helpful and accommodating while managing expectations
- Always mention the booking process"""
    
    def _generate_mock_slots(self) -> List[Dict[str, Any]]:
        """Generate mock appointment slots for demo"""
        slots = []
        base_date = datetime.now() + timedelta(days=1)
        
        for day_offset in range(14):  # Next 2 weeks
            date = base_date + timedelta(days=day_offset)
            
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            # Morning slots
            for hour in [9, 10, 11]:
                slots.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "time": f"{hour:02d}:00",
                    "provider": "Dr. Smith" if hour % 2 == 0 else "Dr. Johnson",
                    "type": "general",
                    "duration_minutes": 60
                })
            
            # Afternoon slots
            for hour in [14, 15, 16]:
                slots.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "time": f"{hour:02d}:00",
                    "provider": "Dr. Smith" if hour % 2 == 0 else "Dr. Johnson",
                    "type": "general",
                    "duration_minutes": 60
                })
        
        return slots
    
    def _get_available_slots(
        self,
        tenant_id: str,
        preferred_dates: Optional[List[str]] = None,
        preferred_times: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available appointment slots
        
        Args:
            tenant_id: Tenant ID
            preferred_dates: Preferred dates
            preferred_times: Preferred times (morning, afternoon, evening)
            
        Returns:
            List of available slots
        """
        
        # Filter mock slots based on preferences
        filtered_slots = self.mock_slots.copy()
        
        # Filter by preferred times
        if preferred_times:
            time_filtered = []
            for slot in filtered_slots:
                hour = int(slot["time"].split(":")[0])
                
                if "morning" in preferred_times and 8 <= hour < 12:
                    time_filtered.append(slot)
                elif "afternoon" in preferred_times and 12 <= hour < 17:
                    time_filtered.append(slot)
                elif "evening" in preferred_times and hour >= 17:
                    time_filtered.append(slot)
            
            filtered_slots = time_filtered if time_filtered else filtered_slots
        
        # Return top slots
        return filtered_slots[:10]
    
    def _format_slots(self, slots: List[Dict[str, Any]]) -> str:
        """Format slots for display"""
        if not slots:
            return "No available slots found."
        
        formatted = []
        for slot in slots[:5]:  # Show first 5
            formatted.append(
                f"- {slot['date']} at {slot['time']} with {slot['provider']} ({slot['duration_minutes']} min)"
            )
        
        return "\n".join(formatted)