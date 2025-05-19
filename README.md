# Multi-Agent Research Assistant

A sophisticated research assistant that leverages multiple AI agents to analyze academic papers and generate structured reports. The system uses advanced RAG (Retrieval-Augmented Generation) techniques and agent coordination to provide comprehensive research insights.

## Features

- Multi-agent system for research paper analysis
- Advanced RAG pipeline with hybrid retrieval
- PDF parsing and document processing
- Structured report generation
- Scalable and modular architecture

## Prerequisites

- Python 3.9+
- Qdrant Cloud account
- OpenAI API key (or other LLM provider)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi_agent_research_assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

```
multi_agent_research_assistant/
├── agents/              # Agent implementations
├── rag/                 # RAG pipeline components
├── graph/              # LangGraph orchestration
├── data/               # Document storage
├── outputs/            # Generated reports
├── tests/              # Test suite
└── utils/              # Utility functions
```

## Usage

1. Place academic papers in the `data/documents/` directory
2. Run the main script:
```bash
python main.py
```

3. Find generated reports in the `outputs/reports/` directory

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- LangChain for RAG orchestration
- LangGraph for agent coordination
- Qdrant for vector storage
- Sentence Transformers for embeddings 