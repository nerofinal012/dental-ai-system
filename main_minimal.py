# main_minimal.py
"""
Minimal working version of Dental AI System
Perfect for demo and free tier API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Dental AI System - Minimal Version")

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("⚠️ Warning: OPENAI_API_KEY not found in .env file")

# Mock document database
MOCK_DOCUMENTS = {
    "office_hours": "Our office hours are Monday-Friday 8AM-5PM, Saturday 9AM-2PM. Closed Sundays.",
    "insurance": "We accept Delta Dental, Aetna, Cigna. Preventive care 100% covered, basic procedures 70-80%, major procedures 50%.",
    "root_canal": "Root canal saves infected teeth. Procedure takes 1-2 visits. Cost: $700-$1500. Insurance covers 50-80%.",
    "emergency": "For emergencies call (555) 123-4567. Same-day appointments when available.",
    "cleaning": "Professional cleaning recommended every 6 months. Takes 30-60 minutes. Usually 100% covered by insurance."
}

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    tenant_id: str = "11111111-1111-1111-1111-111111111111"
    user_id: str = "demo_user"
    user_role: str = "patient"

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    trace_id: str

# Simple search function
def search_documents(query: str) -> List[Dict[str, Any]]:
    """Simple keyword search in mock documents"""
    results = []
    query_lower = query.lower()
    
    for doc_id, content in MOCK_DOCUMENTS.items():
        if any(word in content.lower() for word in query_lower.split()):
            results.append({
                "doc_id": doc_id,
                "content": content,
                "score": 0.8  # Mock score
            })
    
    return results[:3]  # Return top 3

# API Endpoints
@app.get("/")
def root():
    return {"message": "Dental AI System - Minimal Version", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "openai_configured": bool(openai.api_key)}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Simple RAG endpoint"""
    
    try:
        # Search for relevant documents
        search_results = search_documents(request.query)
        
        if not search_results:
            return QueryResponse(
                answer="I couldn't find information about that in our documentation.",
                citations=[],
                confidence=0.0,
                trace_id="demo_001"
            )
        
        # Build context from search results
        context = "\n".join([r["content"] for r in search_results])
        
        # Call OpenAI (or return mock response if no API key)
        if openai.api_key:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for free tier
                messages=[
                    {"role": "system", "content": "You are a helpful dental assistant. Answer based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
                ],
                max_tokens=200,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        else:
            # Mock response if no API key
            answer = f"Based on our records: {search_results[0]['content']}"
        
        # Prepare citations
        citations = [
            {"doc_id": r["doc_id"], "text": r["content"][:100] + "..."}
            for r in search_results
        ]
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=0.85,
            trace_id="demo_001"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent")
async def agent_task(request: Dict[str, Any]):
    """Simplified multi-agent endpoint"""
    
    task = request.get("task", "")
    
    # Simple task routing
    if "appointment" in task.lower() or "schedule" in task.lower():
        result = {
            "agent": "scheduler",
            "response": "Available appointments: Tomorrow 2PM, Thursday 10AM, Friday 3PM. Call (555) 123-4567 to book."
        }
    elif "insurance" in task.lower() or "cost" in task.lower():
        result = {
            "agent": "billing",
            "response": "Insurance typically covers: Preventive 100%, Basic 70-80%, Major 50%. We accept most major plans."
        }
    else:
        result = {
            "agent": "retriever",
            "response": "I'll help you with that. Please be more specific about what you need."
        }
    
    return {
        "result": result,
        "trace": [result],
        "total_tokens": 100,
        "total_cost": 0.002
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Dental AI System (Minimal Version)...")
    print("📝 OpenAI API Key:", "Configured ✅" if openai.api_key else "Not found ❌")
    print("🌐 API Docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)