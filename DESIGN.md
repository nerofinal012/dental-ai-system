# System Design Document

## Architecture Overview

The Dental AI System uses a multi-agent RAG architecture with the following key components:

### Core Architecture
- **FastAPI Backend**: Handles HTTP requests and orchestrates agents
- **Multi-Agent System**: Specialized agents for different domains
- **Hybrid RAG**: Combines keyword and semantic search
- **Frontend**: Modern web interface with real-time chat

## Key Design Decisions

### 1. Multi-Agent vs Single LLM
**Decision**: Use specialized agents instead of a single LLM
**Rationale**: 
- Better accuracy for domain-specific tasks
- Easier debugging and monitoring
- Cost optimization (can use smaller models for simple tasks)
**Trade-off**: Increased complexity vs better specialization

### 2. Hybrid Search Strategy
**Decision**: Combine keyword matching with vector search
**Rationale**: 
- Keywords catch exact matches (phone numbers, hours)
- Vectors capture semantic similarity
**Trade-off**: Slightly higher latency vs better accuracy

### 3. In-Memory Storage (for MVP)
**Decision**: Use in-memory document storage instead of PostgreSQL
**Rationale**: 
- Faster development for proof of concept
- No database setup required for demo
**Trade-off**: Not production-ready vs rapid prototyping

### 4. PHI Protection Strategy
**Decision**: Regex-based detection and redaction
**Rationale**: 
- Fast and reliable for common patterns
- No external dependencies
**Trade-off**: May miss complex PHI vs quick implementation

## Agent Roles

1. **Planner**: Decomposes complex queries into subtasks
2. **Retriever**: Searches documents and provides citations
3. **Scheduler**: Manages appointment logic
4. **Safety**: PHI redaction and content filtering
5. **Summarizer**: Consolidates multi-agent outputs

## Security Model

- **Tenant Isolation**: Every query filtered by tenant_id
- **RBAC**: Patient, Staff, Admin roles with different permissions
- **PHI Protection**: Automatic detection and redaction
- **Input Validation**: Both client and server side

## Performance Targets

- Response time: <500ms P50, <1s P95
- Accuracy: >90% for standard queries
- Token usage: <1000 per typical query

## Future Improvements

1. PostgreSQL with pgvector for production
2. Redis caching for frequent queries
3. Real appointment database integration
4. Advanced reranking models
5. Streaming responses for better UX