# 🦷 Dental Practice Multi-Agent RAG System

An AI-powered dental practice assistant that uses multi-agent orchestration and RAG (Retrieval-Augmented Generation) to handle patient inquiries, appointment scheduling, and insurance information.

## Features

- Multi-Agent Architecture: Specialized agents for different tasks (Planner, Retriever, Scheduler, Safety, Summarizer)
- RAG Implementation: Hybrid search combining vector and keyword matching
- PHI Protection: Automatic detection and redaction of sensitive information
- Multi-Tenant Support: Secure tenant isolation for multiple practices
- Modern Web Interface: Interactive chat interface with citation display

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

1. Clone the repository
2. Create virtual environment: python -m venv venv
3. Activate it: venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux)
4. Install dependencies: pip install -r requirements.txt
5. Copy .env.example to .env and add your OpenAI API key
6. Run: python main.py
7. Open frontend/index.html in your browser

## Project Structure

- main.py - FastAPI backend
- agents/ - Multi-agent system
- rag/ - RAG implementation  
- frontend/ - Web interface
- evaluation/ - Testing framework

## API Endpoints

- GET / - Root endpoint
- GET /health - Health check
- POST /ask - Single query with RAG
- POST /agent - Multi-agent task
- GET /metrics - Performance metrics

## Author

nerofinal012
