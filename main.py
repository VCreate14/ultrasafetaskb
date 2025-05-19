import os
import asyncio
from dotenv import load_dotenv
from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent
from agents.writer_agent import WriterAgent
from graph.coordinator import ResearchCoordinator
from rag.embeddings import EmbeddingsManager
from rag.retriever import HybridRetriever
from utils.pdf_parser import PDFParser

async def process_research_query(coordinator: ResearchCoordinator, query: str) -> None:
    """Process a single research query and display results.
    
    Args:
        coordinator: ResearchCoordinator instance
        query: Research query to process
    """
    print(f"\nProcessing research query: {query}")
    print("Starting research workflow...")
    
    try:
        # Process query through workflow
        results = await coordinator.process_query(query)
        
        # Print results
        print("\nResearch Results:")
        print(f"Number of documents found: {len(results['documents'])}")
        print(f"Number of summaries generated: {len(results['summaries'])}")
        
        if results['errors']:
            print("\nErrors encountered:")
            for error in results['errors']:
                print(f"- {error}")
        
        if results['report']:
            print("\nReport generated successfully!")
            print(f"Report saved to: {results['report'].get('file_path', 'N/A')}")
            
            # Print executive summary if available
            if 'executive_summary' in results['report']:
                print("\nExecutive Summary:")
                print(results['report']['executive_summary'])
    
    except Exception as e:
        print(f"\nError processing query: {str(e)}")

async def main():
    """Main entry point for the research assistant."""
    # Load environment variables
    load_dotenv()
    
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
    
    print("Welcome to the Research Assistant!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'help' for available commands.")
    
    while True:
        try:
            # Get user input
            query = input("\nEnter your research query: ").strip()
            
            # Check for exit command
            if query.lower() in ['exit', 'quit']:
                print("\nThank you for using the Research Assistant. Goodbye!")
                break
            
            # Check for help command
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("- Type your research query to start a new research task")
                print("- Type 'exit' or 'quit' to end the session")
                print("- Type 'help' to show this help message")
                continue
            
            # Skip empty queries
            if not query:
                print("Please enter a valid research query.")
                continue
            
            # Process the query
            await process_research_query(coordinator, query)
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Type 'exit' to quit or continue with a new query.")
            continue
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nResearch Assistant terminated. Goodbye!") 