from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json
import os

from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent
from agents.writer_agent import WriterAgent
from graph.coordinator import ResearchCoordinator
from rag.embeddings import EmbeddingsManager
from rag.retriever import HybridRetriever
from utils.pdf_parser import PDFParser

# Initialize FastAPI app
app = FastAPI(
    title="Research Assistant API",
    description="API for multi-agent research assistant system",
    version="0.1.0"
)

# Initialize components
embeddings_manager = EmbeddingsManager()
retriever = HybridRetriever(embeddings_manager)
pdf_parser = PDFParser()

# Initialize agents
research_agent = ResearchAgent(retriever, pdf_parser)
summarizer_agent = SummarizerAgent()
critic_agent = CriticAgent()
writer_agent = WriterAgent()

# Initialize coordinator
coordinator = ResearchCoordinator(
    research_agent=research_agent,
    summarizer_agent=summarizer_agent,
    critic_agent=critic_agent,
    writer_agent=writer_agent
)

# Request/Response Models
class ResearchQuery(BaseModel):
    query: str
    max_documents: Optional[int] = 10
    min_relevance_score: Optional[float] = 0.5

class ResearchResponse(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    evaluations: Dict[str, Any]
    synthesis: Dict[str, Any]
    report: Dict[str, Any]
    errors: List[str]
    timestamp: str

# Store for research results
research_results = {}

async def process_research_task(query_id: str, query: str):
    """Background task to process research query."""
    try:
        results = await coordinator.process_query(query)
        results["timestamp"] = datetime.now().isoformat()
        research_results[query_id] = results
    except Exception as e:
        research_results[query_id] = {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/research", response_model=Dict[str, str])
async def start_research(query: ResearchQuery, background_tasks: BackgroundTasks):
    """Start a research task."""
    query_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    background_tasks.add_task(process_research_task, query_id, query.query)
    return {"query_id": query_id, "status": "processing"}

@app.get("/research/{query_id}", response_model=ResearchResponse)
async def get_research_results(query_id: str):
    """Get research results for a query ID."""
    if query_id not in research_results:
        raise HTTPException(status_code=404, detail="Research task not found")
    return research_results[query_id]

@app.get("/research/{query_id}/status")
async def get_research_status(query_id: str):
    """Get the status of a research task."""
    if query_id not in research_results:
        raise HTTPException(status_code=404, detail="Research task not found")
    return {
        "query_id": query_id,
        "status": "completed" if "error" not in research_results[query_id] else "failed",
        "timestamp": research_results[query_id]["timestamp"]
    }

@app.get("/workflow/status")
async def get_workflow_status():
    """Get the current status of the workflow."""
    return coordinator.get_workflow_status()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat()
    } 