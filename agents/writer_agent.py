from typing import List, Dict, Any
from langchain.schema import Document
from .base_agent import BaseAgent
from llm.ultrasafe_llm import UltraSafeLLM
from config import settings
import json
import os
from datetime import datetime

class WriterAgent(BaseAgent):
    """Agent responsible for compiling research findings into reports."""
    
    def __init__(self, output_dir: str = None):
        super().__init__(
            name="WriterAgent",
            description="Compiles research findings into structured reports"
        )
        self.llm = UltraSafeLLM(
            api_key=settings.ULTRASAFE_API_KEY,
            model=settings.ULTRASAFE_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        self.output_dir = output_dir or settings.DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def process(
        self,
        query: str,
        documents: List[Document],
        summaries: List[Dict[str, Any]],
        evaluations: Dict[str, Any],
        synthesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile research findings into a report.
        
        Args:
            query: Original research query
            documents: List of documents
            summaries: List of document summaries
            evaluations: Document evaluations
            synthesis: Cross-document synthesis
            
        Returns:
            Generated report
        """
        try:
            # Generate report sections
            executive_summary = await self._generate_executive_summary(
                query, summaries, evaluations, synthesis
            )
            
            methodology = await self._generate_methodology_section(
                documents, evaluations
            )
            
            findings = await self._generate_findings_section(
                summaries, synthesis
            )
            
            analysis = await self._generate_analysis_section(
                evaluations, synthesis
            )
            
            recommendations = await self._generate_recommendations(
                synthesis, evaluations
            )
            
            # Compile full report
            report = {
                "title": f"Research Report: {query}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "executive_summary": executive_summary,
                "methodology": methodology,
                "findings": findings,
                "analysis": analysis,
                "recommendations": recommendations,
                "references": self._generate_references(documents)
            }
            
            # Save report
            report_path = await self._save_report(report)
            
            # Update agent state
            self.update_state(context={"report": report, "report_path": report_path})
            
            return report
            
        except Exception as e:
            self.add_error(f"Error generating report: {str(e)}")
            return {}
    
    async def _generate_executive_summary(
        self,
        query: str,
        summaries: List[Dict[str, Any]],
        evaluations: Dict[str, Any],
        synthesis: Dict[str, Any]
    ) -> str:
        """Generate executive summary.
        
        Args:
            query: Research query
            summaries: Document summaries
            evaluations: Document evaluations
            synthesis: Cross-document synthesis
            
        Returns:
            Executive summary text
        """
        try:
            prompt = f"""
            Generate an executive summary for this research report:
            
            Query: {query}
            
            Key Findings:
            {json.dumps(synthesis, indent=2)}
            
            Document Quality:
            {json.dumps(evaluations['average_scores'], indent=2)}
            
            Focus on:
            1. Main research question
            2. Key findings
            3. Quality of evidence
            4. Major implications
            5. Recommendations
            
            Write in a clear, concise style suitable for executive readers.
            """
            
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            self.add_error(f"Error generating executive summary: {str(e)}")
            return ""
    
    async def _generate_methodology_section(
        self,
        documents: List[Document],
        evaluations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate methodology section.
        
        Args:
            documents: List of documents
            evaluations: Document evaluations
            
        Returns:
            Methodology section content
        """
        try:
            prompt = f"""
            Generate a methodology section for this research report:
            
            Documents Analyzed: {len(documents)}
            Quality Scores: {json.dumps(evaluations['average_scores'], indent=2)}
            
            Include:
            1. Research approach
            2. Document selection criteria
            3. Analysis methods
            4. Quality assessment
            5. Limitations
            
            Format as a structured dictionary.
            """
            
            response = await self.llm.agenerate([prompt])
            return eval(response.generations[0][0].text)
            
        except Exception as e:
            self.add_error(f"Error generating methodology section: {str(e)}")
            return {}
    
    async def _generate_findings_section(
        self,
        summaries: List[Dict[str, Any]],
        synthesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate findings section.
        
        Args:
            summaries: Document summaries
            synthesis: Cross-document synthesis
            
        Returns:
            Findings section content
        """
        try:
            prompt = f"""
            Generate a findings section for this research report:
            
            Document Summaries:
            {json.dumps(summaries, indent=2)}
            
            Synthesis:
            {json.dumps(synthesis, indent=2)}
            
            Include:
            1. Key findings by theme
            2. Supporting evidence
            3. Contradictory findings
            4. Emerging patterns
            5. Knowledge gaps
            
            Format as a structured dictionary.
            """
            
            response = await self.llm.agenerate([prompt])
            return eval(response.generations[0][0].text)
            
        except Exception as e:
            self.add_error(f"Error generating findings section: {str(e)}")
            return {}
    
    async def _generate_analysis_section(
        self,
        evaluations: Dict[str, Any],
        synthesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analysis section.
        
        Args:
            evaluations: Document evaluations
            synthesis: Cross-document synthesis
            
        Returns:
            Analysis section content
        """
        try:
            prompt = f"""
            Generate an analysis section for this research report:
            
            Evaluations:
            {json.dumps(evaluations, indent=2)}
            
            Synthesis:
            {json.dumps(synthesis, indent=2)}
            
            Include:
            1. Critical analysis of findings
            2. Quality assessment
            3. Reliability of evidence
            4. Implications
            5. Future research needs
            
            Format as a structured dictionary.
            """
            
            response = await self.llm.agenerate([prompt])
            return eval(response.generations[0][0].text)
            
        except Exception as e:
            self.add_error(f"Error generating analysis section: {str(e)}")
            return {}
    
    async def _generate_recommendations(
        self,
        synthesis: Dict[str, Any],
        evaluations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations.
        
        Args:
            synthesis: Cross-document synthesis
            evaluations: Document evaluations
            
        Returns:
            List of recommendations
        """
        try:
            prompt = f"""
            Generate recommendations based on this research:
            
            Synthesis:
            {json.dumps(synthesis, indent=2)}
            
            Quality Assessment:
            {json.dumps(evaluations['average_scores'], indent=2)}
            
            Include:
            1. Research recommendations
            2. Practical implications
            3. Policy recommendations
            4. Implementation suggestions
            5. Future directions
            
            Format as a list of dictionaries with 'category' and 'recommendation' fields.
            """
            
            response = await self.llm.agenerate([prompt])
            return eval(response.generations[0][0].text)
            
        except Exception as e:
            self.add_error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _generate_references(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Generate reference list.
        
        Args:
            documents: List of documents
            
        Returns:
            List of references
        """
        try:
            references = []
            for doc in documents:
                ref = {
                    "title": doc.metadata.get("title", ""),
                    "authors": doc.metadata.get("authors", []),
                    "year": doc.metadata.get("year", ""),
                    "source": doc.metadata.get("source", ""),
                    "url": doc.metadata.get("url", "")
                }
                references.append(ref)
            return references
            
        except Exception as e:
            self.add_error(f"Error generating references: {str(e)}")
            return []
    
    async def _save_report(self, report: Dict[str, Any]) -> str:
        """Save report to file.
        
        Args:
            report: Generated report
            
        Returns:
            Path to saved report
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save report
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            return filepath
            
        except Exception as e:
            self.add_error(f"Error saving report: {str(e)}")
            return "" 