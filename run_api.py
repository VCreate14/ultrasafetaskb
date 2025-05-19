import uvicorn
from dotenv import load_dotenv

def main():
    """Run the FastAPI server."""
    # Load environment variables
    load_dotenv()
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 