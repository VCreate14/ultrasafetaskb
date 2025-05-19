from typing import Dict, List, Any, Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent
from agents.writer_agent import WriterAgent
from config import settings

class ResearchState(TypedDict):
    """State for the research workflow."""
    query: str
    documents: List[Any]
    summaries: List[Dict[str, Any]]
    evaluations: Dict[str, Any]
    synthesis: Dict[str, Any]
    report: Dict[str, Any]
    errors: List[str]

class ResearchCoordinator:
    """Coordinates the research workflow using LangGraph."""
    
    def __init__(
        self,
        research_agent: Optional[ResearchAgent] = None,
        summarizer_agent: Optional[SummarizerAgent] = None,
        critic_agent: Optional[CriticAgent] = None,
        writer_agent: Optional[WriterAgent] = None
    ):
        # Initialize agents with provided instances or create new ones
        self.research_agent = research_agent or ResearchAgent()
        self.summarizer_agent = summarizer_agent or SummarizerAgent()
        self.critic_agent = critic_agent or CriticAgent()
        self.writer_agent = writer_agent or WriterAgent()
        
        # Create and compile the graph
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """Create the research workflow graph."""
        # Create the graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("write", self._write_node)
        
        # Add edges
        workflow.add_edge("research", "summarize")
        workflow.add_edge("summarize", "evaluate")
        workflow.add_edge("evaluate", "synthesize")
        workflow.add_edge("synthesize", "write")
        workflow.add_edge("write", END)
        
        # Set entry point
        workflow.set_entry_point("research")
        
        return workflow
    
    async def _research_node(self, state: ResearchState) -> ResearchState:
        """Research node that finds relevant documents."""
        try:
            documents = await self.research_agent.process(state["query"])
            return {**state, "documents": documents}
        except Exception as e:
            return {**state, "errors": state.get("errors", []) + [str(e)]}
    
    async def _summarize_node(self, state: ResearchState) -> ResearchState:
        """Summarize node that extracts key information."""
        try:
            summaries = await self.summarizer_agent.process(state["documents"])
            return {**state, "summaries": summaries}
        except Exception as e:
            return {**state, "errors": state.get("errors", []) + [str(e)]}
    
    async def _evaluate_node(self, state: ResearchState) -> ResearchState:
        """Evaluate node that assesses document quality."""
        try:
            evaluations = await self.critic_agent.process(
                state["documents"],
                state["summaries"]
            )
            return {**state, "evaluations": evaluations}
        except Exception as e:
            return {**state, "errors": state.get("errors", []) + [str(e)]}
    
    async def _synthesize_node(self, state: ResearchState) -> ResearchState:
        """Synthesize node that combines insights."""
        try:
            synthesis = await self.summarizer_agent.cross_document_synthesis(
                state["summaries"]
            )
            return {**state, "synthesis": synthesis}
        except Exception as e:
            return {**state, "errors": state.get("errors", []) + [str(e)]}
    
    async def _write_node(self, state: ResearchState) -> ResearchState:
        """Write node that generates the final report."""
        try:
            report = await self.writer_agent.process(
                state["query"],
                state["documents"],
                state["summaries"],
                state["evaluations"],
                state["synthesis"]
            )
            return {**state, "report": report}
        except Exception as e:
            return {**state, "errors": state.get("errors", []) + [str(e)]}
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a research query through the workflow.
        
        Args:
            query: Research query to process
            
        Returns:
            Dictionary containing the results of the workflow
        """
        # Initialize state
        initial_state = {
            "query": query,
            "documents": [],
            "summaries": [],
            "evaluations": {},
            "synthesis": {},
            "report": {},
            "errors": []
        }
        
        # Run the workflow using the compiled app
        final_state = await self.app.ainvoke(initial_state)
        
        return final_state 