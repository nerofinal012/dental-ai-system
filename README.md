# Dental Practice Multi-Agent RAG System

An AI-powered dental practice assistant that uses multi-agent orchestration and RAG (Retrieval-Augmented Generation) to handle patient inquiries, appointment scheduling, and insurance information.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks (Planner, Retriever, Scheduler, Safety, Summarizer)
- **RAG Implementation**: Hybrid search combining vector and keyword matching
- **PHI Protection**: Automatic detection and redaction of sensitive information
- **Multi-Tenant Support**: Secure tenant isolation for multiple practices
- **Modern Web Interface**: Interactive chat interface with citation display

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nerofinal012/dental-ai-system.git
cd dental-ai-system