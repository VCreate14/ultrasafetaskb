import os
from typing import Dict, Any, Optional
import PyPDF2
from langchain.schema import Document

class PDFParser:
    """Utility for parsing PDF documents."""
    
    def __init__(self):
        """Initialize the PDF parser."""
        pass
    
    async def process(self, content: str) -> str:
        """Process PDF content and extract text.
        
        Args:
            content: PDF content as string
            
        Returns:
            Extracted text
        """
        try:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(content)
            
            # Extract text from each page
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF content: {str(e)}")
    
    async def process_file(self, file_path: str) -> Document:
        """Process a PDF file and return a Document object.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document object containing the extracted text and metadata
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            metadata = self._extract_metadata_from_filename(filename)
            
            # Read and process the PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Create document
                return Document(
                    page_content=text.strip(),
                    metadata=metadata
                )
                
        except Exception as e:
            raise Exception(f"Error processing PDF file: {str(e)}")
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename.
        
        Args:
            filename: PDF filename
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Remove file extension
            name = os.path.splitext(filename)[0]
            
            # Split filename into components
            components = name.split('_')
            
            # Extract metadata
            metadata = {
                "title": components[0] if components else "",
                "authors": components[1].split('-') if len(components) > 1 else [],
                "year": components[2] if len(components) > 2 else "",
                "file_type": "pdf",
                "filename": filename
            }
            
            return metadata
        except Exception as e:
            return {
                "title": filename,
                "authors": [],
                "year": "",
                "file_type": "pdf",
                "filename": filename
            }
    
    async def extract_metadata_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata directly from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get document info
                info = pdf_reader.metadata
                
                metadata = {
                    "title": info.get('/Title', ''),
                    "authors": info.get('/Author', '').split(',') if info.get('/Author') else [],
                    "year": info.get('/CreationDate', '')[:4] if info.get('/CreationDate') else '',
                    "file_type": "pdf",
                    "filename": os.path.basename(file_path)
                }
                
                return metadata
        except Exception as e:
            return self._extract_metadata_from_filename(os.path.basename(file_path)) 