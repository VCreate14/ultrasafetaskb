from typing import List, Dict, Any
from langchain.schema import Document
from .base_agent import BaseAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from llm.ultrasafe_llm import UltraSafeLLM
from config import settings

class SummarizerAgent(BaseAgent):
    """Agent responsible for summarizing academic papers."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(
            name="SummarizerAgent",
            description="Extracts key information from academic papers"
        )
        self.chunk_size = chunk_size or settings.MAX_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.llm = UltraSafeLLM(
            api_key=settings.ULTRASAFE_API_KEY,
            model=settings.ULTRASAFE_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        self.summarize_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            verbose=True
        )
    
    async def process(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Process documents and extract key information.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of summaries with key information
        """
        try:
            summaries = []
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_documents([doc])
                
                # Generate summary
                summary = await self.summarize_chain.arun(chunks)
                
                # Extract key information
                key_info = await self._extract_key_information(doc, summary)
                
                summaries.append({
                    "title": doc.metadata.get("title", ""),
                    "summary": summary,
                    "key_findings": key_info,
                    "metadata": doc.metadata
                })
            
            # Update agent state
            self.update_state(context={"summaries": summaries})
            
            return summaries
            
        except Exception as e:
            self.add_error(f"Error summarizing documents: {str(e)}")
            return []
    
    async def _extract_key_information(self, doc: Document, summary: str) -> Dict[str, Any]:
        """Extract key information from document and summary.
        
        Args:
            doc: Original document
            summary: Generated summary
            
        Returns:
            Dictionary containing key information
        """
        try:
            # Extract key findings using LLM
            prompt = f"""
            Extract key findings from the following academic paper summary:
            
            Summary: {summary}
            
            Extract:
            1. Main research question
            2. Key methodologies
            3. Major findings
            4. Limitations
            5. Future work
            
            Format as a structured dictionary.
            """
            
            response = await self.llm.agenerate([prompt])
            key_info = eval(response.generations[0][0].text)
            
            return key_info
            
        except Exception as e:
            self.add_error(f"Error extracting key information: {str(e)}")
            return {
                "main_question": "",
                "methodologies": [],
                "findings": [],
                "limitations": [],
                "future_work": []
            }
    
    async def cross_document_synthesis(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize information across multiple documents.
        
        Args:
            summaries: List of document summaries
            
        Returns:
            Synthesized information
        """
        try:
            # Combine summaries for synthesis
            combined_text = "\n\n".join([
                f"Title: {s['title']}\nSummary: {s['summary']}"
                for s in summaries
            ])
            
            # Generate synthesis using LLM
            prompt = f"""
            Synthesize key insights from the following academic papers:
            
            {combined_text}
            
            Provide:
            1. Common themes
            2. Conflicting findings
            3. Complementary insights
            4. Research gaps
            5. Future directions
            
            Format as a structured dictionary.
            """
            
            response = await self.llm.agenerate([prompt])
            synthesis = eval(response.generations[0][0].text)
            
            return synthesis
            
        except Exception as e:
            self.add_error(f"Error synthesizing documents: {str(e)}")
            return {
                "common_themes": [],
                "conflicting_findings": [],
                "complementary_insights": [],
                "research_gaps": [],
                "future_directions": []
            } 