from typing import List, Dict, Any
from langchain.schema import Document
from .base_agent import BaseAgent
from llm.ultrasafe_llm import UltraSafeLLM
from config import settings

class CriticAgent(BaseAgent):
    """Agent responsible for evaluating information quality and relevance."""
    
    def __init__(self):
        super().__init__(
            name="CriticAgent",
            description="Evaluates the quality and relevance of information"
        )
        self.llm = UltraSafeLLM(
            api_key=settings.ULTRASAFE_API_KEY,
            model=settings.ULTRASAFE_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
    
    async def process(self, documents: List[Document], summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate documents and their summaries.
        
        Args:
            documents: List of original documents
            summaries: List of document summaries
            
        Returns:
            Evaluation results
        """
        try:
            evaluations = []
            for doc, summary in zip(documents, summaries):
                # Evaluate document quality
                quality_score = await self._evaluate_quality(doc, summary)
                
                # Evaluate relevance
                relevance_score = await self._evaluate_relevance(doc, summary)
                
                # Evaluate methodology
                methodology_score = await self._evaluate_methodology(doc)
                
                evaluations.append({
                    "title": doc.metadata.get("title", ""),
                    "quality_score": quality_score,
                    "relevance_score": relevance_score,
                    "methodology_score": methodology_score,
                    "overall_score": (quality_score + relevance_score + methodology_score) / 3,
                    "critique": await self._generate_critique(doc, summary)
                })
            
            # Update agent state
            self.update_state(context={"evaluations": evaluations})
            
            return {
                "evaluations": evaluations,
                "average_scores": self._calculate_average_scores(evaluations)
            }
            
        except Exception as e:
            self.add_error(f"Error evaluating documents: {str(e)}")
            return {"evaluations": [], "average_scores": {}}
    
    async def _evaluate_quality(self, doc: Document, summary: Dict[str, Any]) -> float:
        """Evaluate the quality of a document and its summary.
        
        Args:
            doc: Original document
            summary: Document summary
            
        Returns:
            Quality score (0-1)
        """
        try:
            prompt = f"""
            Evaluate the quality of this academic paper and its summary:
            
            Title: {doc.metadata.get('title', '')}
            Content: {doc.page_content[:1000]}...
            Summary: {summary['summary']}
            
            Consider:
            1. Clarity of writing
            2. Logical flow
            3. Evidence quality
            4. Citation quality
            5. Summary accuracy
            
            Provide a score from 0 to 1.
            """
            
            response = await self.llm.agenerate([prompt])
            score = float(response.generations[0][0].text.strip())
            return min(max(score, 0), 1)
            
        except Exception as e:
            self.add_error(f"Error evaluating quality: {str(e)}")
            return 0.0
    
    async def _evaluate_relevance(self, doc: Document, summary: Dict[str, Any]) -> float:
        """Evaluate the relevance of a document.
        
        Args:
            doc: Original document
            summary: Document summary
            
        Returns:
            Relevance score (0-1)
        """
        try:
            prompt = f"""
            Evaluate the relevance of this academic paper:
            
            Title: {doc.metadata.get('title', '')}
            Summary: {summary['summary']}
            
            Consider:
            1. Topic relevance
            2. Timeliness
            3. Impact potential
            4. Field significance
            
            Provide a score from 0 to 1.
            """
            
            response = await self.llm.agenerate([prompt])
            score = float(response.generations[0][0].text.strip())
            return min(max(score, 0), 1)
            
        except Exception as e:
            self.add_error(f"Error evaluating relevance: {str(e)}")
            return 0.0
    
    async def _evaluate_methodology(self, doc: Document) -> float:
        """Evaluate the methodology of a document.
        
        Args:
            doc: Original document
            
        Returns:
            Methodology score (0-1)
        """
        try:
            prompt = f"""
            Evaluate the methodology of this academic paper:
            
            Title: {doc.metadata.get('title', '')}
            Content: {doc.page_content[:1000]}...
            
            Consider:
            1. Research design
            2. Data collection
            3. Analysis methods
            4. Validity
            5. Reproducibility
            
            Provide a score from 0 to 1.
            """
            
            response = await self.llm.agenerate([prompt])
            score = float(response.generations[0][0].text.strip())
            return min(max(score, 0), 1)
            
        except Exception as e:
            self.add_error(f"Error evaluating methodology: {str(e)}")
            return 0.0
    
    async def _generate_critique(self, doc: Document, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed critique of the document.
        
        Args:
            doc: Original document
            summary: Document summary
            
        Returns:
            Dictionary containing critique
        """
        try:
            prompt = f"""
            Provide a detailed critique of this academic paper:
            
            Title: {doc.metadata.get('title', '')}
            Summary: {summary['summary']}
            
            Include:
            1. Strengths
            2. Weaknesses
            3. Methodological concerns
            4. Contribution to field
            5. Recommendations
            
            Format as a structured dictionary.
            """
            
            response = await self.llm.agenerate([prompt])
            critique = eval(response.generations[0][0].text)
            return critique
            
        except Exception as e:
            self.add_error(f"Error generating critique: {str(e)}")
            return {
                "strengths": [],
                "weaknesses": [],
                "methodological_concerns": [],
                "contribution": "",
                "recommendations": []
            }
    
    def _calculate_average_scores(self, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average scores across all evaluations.
        
        Args:
            evaluations: List of evaluations
            
        Returns:
            Dictionary of average scores
        """
        try:
            if not evaluations:
                return {}
            
            total_quality = sum(e["quality_score"] for e in evaluations)
            total_relevance = sum(e["relevance_score"] for e in evaluations)
            total_methodology = sum(e["methodology_score"] for e in evaluations)
            total_overall = sum(e["overall_score"] for e in evaluations)
            
            n = len(evaluations)
            return {
                "average_quality": total_quality / n,
                "average_relevance": total_relevance / n,
                "average_methodology": total_methodology / n,
                "average_overall": total_overall / n
            }
            
        except Exception as e:
            self.add_error(f"Error calculating average scores: {str(e)}")
            return {} 