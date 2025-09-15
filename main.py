# main.py
"""
Dental Practice Multi-Agent RAG System
Hybrid version - works without Docker/PostgreSQL
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import logging
import json
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dental Practice AI Assistant",
    description="Multi-agent RAG system for dental practice automation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    logger.warning("⚠️ OPENAI_API_KEY not found - using mock responses")

# ==================== Mock Data Store ====================

MOCK_DOCUMENTS = [
    {
        "chunk_id": "chunk_001",
        "doc_id": "doc_001",
        "content": "Our dental practice office hours are Monday through Friday from 8:00 AM to 5:00 PM, and Saturday from 9:00 AM to 2:00 PM. We are closed on Sundays and major holidays.",
        "doc_title": "Office Hours Policy",
        "doc_type": "policy",
        "tenant_id": "11111111-1111-1111-1111-111111111111"
    },
    {
        "chunk_id": "chunk_002",
        "doc_id": "doc_002",
        "content": "We accept most major dental insurance plans including Delta Dental, Aetna, Cigna, and Blue Cross Blue Shield. Preventive care is typically covered at 100%, basic procedures at 70-80%, and major procedures at 50%.",
        "doc_title": "Insurance Coverage",
        "doc_type": "policy",
        "tenant_id": "11111111-1111-1111-1111-111111111111"
    },
    {
        "chunk_id": "chunk_003",
        "doc_id": "doc_003",
        "content": "Root canal treatment is a procedure to repair and save a badly damaged or infected tooth. The procedure typically takes 1-2 visits. Cost ranges from $700-$1500. Most insurance plans cover 50-80% of root canal procedures.",
        "doc_title": "Root Canal Treatment",
        "doc_type": "procedure",
        "tenant_id": "11111111-1111-1111-1111-111111111111"
    },
    {
        "chunk_id": "chunk_004",
        "doc_id": "doc_004",
        "content": "For dental emergencies, we offer same-day appointments when available. Please call our emergency line at (555) 123-4567. After hours, our answering service will connect you with the on-call dentist.",
        "doc_title": "Emergency Care",
        "doc_type": "faq",
        "tenant_id": "11111111-1111-1111-1111-111111111111"
    },
    {
        "chunk_id": "chunk_005",
        "doc_id": "doc_005",
        "content": "Professional teeth cleaning is recommended every six months for most patients. The appointment typically takes 30-60 minutes. Regular cleanings help prevent cavities and gum disease.",
        "doc_title": "Teeth Cleaning",
        "doc_type": "procedure",
        "tenant_id": "11111111-1111-1111-1111-111111111111"
    }
]

# ==================== Request/Response Models ====================

class QueryRequest(BaseModel):
    """Request model for single query"""
    query: str = Field(..., description="User's question")
    tenant_id: str = Field(default="11111111-1111-1111-1111-111111111111")
    user_id: str = Field(default="anonymous")
    user_role: str = Field(default="patient", description="patient|staff|admin")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Citation(BaseModel):
    """Citation model for source attribution"""
    doc_id: str
    chunk_id: str
    text: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    """Response model for single query"""
    answer: str
    citations: List[Citation]
    confidence: float
    trace_id: str
    metrics: Dict[str, Any]

class AgentRequest(BaseModel):
    """Request model for multi-agent tasks"""
    task: str = Field(..., description="Complex task description")
    tenant_id: str = Field(default="11111111-1111-1111-1111-111111111111")
    user_id: str = Field(default="anonymous")
    user_role: str = Field(default="patient")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    """Response model for multi-agent tasks"""
    result: Dict[str, Any]
    agent_trace: List[Dict[str, Any]]
    total_tokens: int
    total_cost: float
    trace_id: str

# ==================== Simple Search & Agents ====================

def search_documents(query: str, tenant_id: str, doc_types: List[str] = None) -> List[Dict[str, Any]]:
    """Enhanced keyword search in mock documents"""
    results = []
    query_words = [word.lower() for word in query.split()]
    
    for doc in MOCK_DOCUMENTS:
        # Filter by tenant
        if doc["tenant_id"] != tenant_id:
            continue
            
        # Filter by doc type if specified
        if doc_types and doc["doc_type"] not in doc_types:
            continue
            
        # Score based on word matches
        content_lower = doc["content"].lower()
        title_lower = doc["doc_title"].lower()
        
        score = 0
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                score += content_lower.count(word) * 1
                score += title_lower.count(word) * 2
        
        if score > 0:
            doc_copy = doc.copy()
            doc_copy["score"] = score / (len(query_words) + 1)
            doc_copy["final_score"] = doc_copy["score"]
            results.append(doc_copy)
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]

def detect_and_redact_phi(text: str) -> tuple[str, bool]:
    """Simple PHI detection and redaction"""
    import re
    
    phi_patterns = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    }
    
    redacted = text
    detected = False
    
    for phi_type, pattern in phi_patterns.items():
        if re.search(pattern, text):
            detected = True
            redacted = re.sub(pattern, f'[{phi_type.upper()}_REDACTED]', redacted)
    
    return redacted, detected

def check_content_safety(text: str) -> tuple[bool, str]:
    """Check if content is appropriate for dental assistant"""
    
    # Inappropriate content patterns
    inappropriate_keywords = [
        'sexual', 'nude', 'porn', 'xxx', 'sex', 'erotic',
        'violence', 'weapon', 'drug', 'illegal',
        'hack', 'exploit', 'injection', 'malware'
    ]
    
    text_lower = text.lower()
    
    for keyword in inappropriate_keywords:
        if keyword in text_lower:
            return False, "I'm a dental practice assistant and can only help with dental-related questions. Please keep our conversation professional and focused on dental health topics."
    
    return True, ""

def check_conversational_intent(text: str) -> tuple[bool, str]:
    """Check if user is trying to have casual conversation"""
    
    text_lower = text.lower().strip()
    
    # Don't treat dental-related questions as conversational
    dental_keywords = ['office', 'hours', 'insurance', 'appointment', 'dental', 'tooth', 'teeth', 
                       'cleaning', 'cavity', 'root canal', 'emergency', 'cost', 'price']
    
    for keyword in dental_keywords:
        if keyword in text_lower:
            return False, ""  # This is a dental query, not casual conversation
    
    # Only handle pure greetings and personal questions
    greetings = ['hi', 'hello', 'hey', 'sup', 'yo']
    if text_lower in greetings:
        return True, "Hello! I'm your AI Dental Assistant. I can help you with scheduling appointments, insurance questions, dental procedures, and emergency care information. What dental-related question can I help you with today?"
    
    # Handle "what/who are you" only as exact matches
    if text_lower == "what are you" or text_lower == "who are you":
        return True, "I'm an AI assistant specialized in dental practice services. I can help you with appointments, insurance, procedures, and dental health questions. What would you like to know?"
    
    # Handle how are you
    if text_lower == "how are you":
        return True, "I'm functioning well and ready to help with your dental needs! How can I assist you with dental services today?"
    
    # Off-topic requests
    offtopic = ['tell me a joke', 'sing a song', 'write a poem', 'play a game']
    if text_lower in offtopic:
        return True, "I'm specifically designed to help with dental-related questions. I can assist with appointments, insurance, procedures, costs, and emergency care. What dental topic can I help you with?"
    
    return False, ""

async def generate_answer(query: str, search_results: List[Dict], use_openai: bool = True) -> str:
    """Generate answer using OpenAI or mock response"""
    
    if not search_results:
        return "I couldn't find specific information about that in our documentation. Please contact our office at (555) 123-4567 for assistance."
    
    # Build context from search results
    context = "\n\n".join([
        f"[{i+1}] {doc['doc_title']}: {doc['content']}"
        for i, doc in enumerate(search_results[:3])
    ])
    
    if use_openai and openai.api_key:
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful dental practice assistant. Answer based only on the provided context. Be concise and helpful."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear answer with citation numbers [1], [2], etc."}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fall back to mock response
    
    # Mock response based on top result
    top_result = search_results[0]
    return f"Based on our {top_result['doc_title']}: {top_result['content'][:200]}... [1]"

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dental Practice AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "operational",
            "openai": "configured" if openai.api_key else "not configured",
            "database": "mock mode"
        }
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Single-turn question answering with RAG grounding"""
    
    trace_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Processing query: {request.query[:50]}... [trace_id: {trace_id}]")
        
        # Step 1: Check content safety
        is_safe, safety_message = check_content_safety(request.query)
        if not is_safe:
            return QueryResponse(
                answer=safety_message,
                citations=[],
                confidence=1.0,
                trace_id=trace_id,
                metrics={"latency_ms": 0, "filtered": True}
            )
        
        # Step 2: Check for conversational intent
        is_conversational, conversational_response = check_conversational_intent(request.query)
        if is_conversational:
            return QueryResponse(
                answer=conversational_response,
                citations=[],
                confidence=1.0,
                trace_id=trace_id,
                metrics={"latency_ms": 0, "conversational": True}
            )
        
        # Step 3: PHI detection and redaction
        safe_query, phi_detected = detect_and_redact_phi(request.query)
        if phi_detected:
            logger.warning(f"PHI detected and redacted in query [trace_id: {trace_id}]")
        
        # Step 4: Search documents
        search_results = search_documents(
            safe_query,
            request.tenant_id,
            ["policy", "procedure", "faq"]
        )
        
        # Step 5: Generate answer
        answer = await generate_answer(safe_query, search_results)
        
        # Step 6: Prepare citations
        citations = [
            Citation(
                doc_id=doc["doc_id"],
                chunk_id=doc["chunk_id"],
                text=doc["content"][:150] + "...",
                relevance_score=doc.get("score", 0.0),
                metadata={"doc_type": doc["doc_type"], "title": doc["doc_title"]}
            )
            for doc in search_results[:3]
        ]
        
        # Calculate confidence
        confidence = min(search_results[0]["score"], 1.0) if search_results else 0.0
        
        # Calculate metrics
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            trace_id=trace_id,
            metrics={
                "latency_ms": elapsed_time * 1000,
                "search_results_count": len(search_results),
                "phi_detected": phi_detected,
                "model_used": "gpt-3.5-turbo" if openai.api_key else "mock"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)} [trace_id: {trace_id}]")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/agent", response_model=AgentResponse)
async def multi_agent_task(request: AgentRequest):
    """Multi-agent orchestration for complex tasks"""
    
    trace_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Processing multi-agent task: {request.task[:50]}... [trace_id: {trace_id}]")
        
        # Simple agent routing based on task keywords
        task_lower = request.task.lower()
        agent_trace = []
        
        # Planner agent
        agent_trace.append({
            "agent": "planner",
            "action": "analyze_task",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Route to appropriate agent
        if any(word in task_lower for word in ["appointment", "schedule", "book", "available"]):
            # Scheduler agent
            result = {
                "agent": "scheduler",
                "response": "I can help you schedule an appointment. We have the following slots available:\n\n• Tomorrow at 2:00 PM with Dr. Smith\n• Thursday at 10:00 AM with Dr. Johnson\n• Friday at 3:00 PM with Dr. Smith\n\nTo book, please call (555) 123-4567 or reply with your preferred time.",
                "available_slots": [
                    {"date": "2024-01-16", "time": "14:00", "provider": "Dr. Smith"},
                    {"date": "2024-01-18", "time": "10:00", "provider": "Dr. Johnson"},
                    {"date": "2024-01-19", "time": "15:00", "provider": "Dr. Smith"}
                ]
            }
            agent_trace.append({
                "agent": "scheduler",
                "action": "check_availability",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif any(word in task_lower for word in ["insurance", "coverage", "cost", "payment"]):
            # Billing agent
            search_results = search_documents("insurance coverage", request.tenant_id, ["policy"])
            answer = await generate_answer(request.task, search_results)
            result = {
                "agent": "billing",
                "response": answer
            }
            agent_trace.append({
                "agent": "billing",
                "action": "check_coverage",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        else:
            # Retriever agent (default)
            search_results = search_documents(request.task, request.tenant_id)
            answer = await generate_answer(request.task, search_results)
            result = {
                "agent": "retriever",
                "response": answer
            }
            agent_trace.append({
                "agent": "retriever",
                "action": "search_documents",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Safety check
        safe_response, phi_detected = detect_and_redact_phi(result["response"])
        if phi_detected:
            result["response"] = safe_response
            agent_trace.append({
                "agent": "safety",
                "action": "redact_phi",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Calculate metrics
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AgentResponse(
            result=result,
            agent_trace=agent_trace,
            total_tokens=len(request.task.split()) * 10,  # Rough estimate
            total_cost=0.001,  # Rough estimate
            trace_id=trace_id
        )
        
    except Exception as e:
        logger.error(f"Error in multi-agent task: {str(e)} [trace_id: {trace_id}]")
        raise HTTPException(status_code=500, detail=f"Error processing task: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """System metrics endpoint"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "requests": {
            "total": 42,
            "per_minute": 2.5,
            "error_rate": 0.02
        },
        "performance": {
            "avg_latency_ms": 250,
            "p95_latency_ms": 450,
            "p99_latency_ms": 780
        },
        "rag": {
            "documents_count": len(MOCK_DOCUMENTS),
            "avg_search_results": 3.2,
            "cache_hit_rate": 0.65
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Dental Practice AI Assistant...")
    print(f"📝 OpenAI API: {'✅ Configured' if openai.api_key else '❌ Not configured (using mock responses)'}")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("💾 Database: Mock mode (no PostgreSQL required)")
    print("-" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")