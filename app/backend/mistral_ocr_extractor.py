"""Module for extracting text from PDF documents using Mistral OCR"""
import os
import logging
from mistralai import Mistral
from s3_utils import upload_pdf_to_s3, upload_file_to_s3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_with_mistral(pdf_path: str) -> str:
    """
    Extract text from a PDF file using Mistral OCR
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        # Get API key from environment
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
            
        # Create client
        client = Mistral(api_key=api_key)
        
        # Read the PDF file and upload to S3
        with open(pdf_path, "rb") as f:
            file_content = f.read()
            document_id = os.path.basename(pdf_path).split('_')[0]  # Extract document_id from filename
            original_filename = os.path.basename(pdf_path)
            
            # Upload to S3 and get URL
            pdf_url = upload_pdf_to_s3(file_content, original_filename, document_id)
            logger.info(f"PDF uploaded to S3: {pdf_url}")
            
        # Process the file with Mistral OCR using the S3 URL
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": pdf_url
            }
        )
            
        # Combine all pages into raw text
        raw_text_parts = []
        for page in ocr_response.pages:
            # Create raw text by removing markdown formatting from the markdown text
            raw_text = page.markdown.replace('#', '').replace('*', '').replace('_', '')
            raw_text_parts.append(raw_text)
        
        raw_text = "\n\n".join(raw_text_parts)
        
        logger.info(f"Successfully extracted {len(raw_text)} characters with Mistral OCR")
        
        return raw_text
            
    except Exception as e:
        logger.error(f"Error processing with Mistral OCR: {str(e)}")
        raise Exception(f"Failed to process PDF with Mistral OCR: {str(e)}")

def process_uploaded_pdf_with_mistral(file_content: bytes, filename: str = "uploaded_file.pdf") -> str:
    """
    Process an uploaded PDF file with Mistral OCR using S3 URL
    
    Args:
        file_content: Binary content of the uploaded PDF file
        filename: Name of the file (for reference purposes)
        
    Returns:
        Extracted text content
    """
    try:
        # Get API key from environment
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
            
        # Create client
        client = Mistral(api_key=api_key)
        
        # Generate a document ID (use current timestamp as part of the ID)
        import uuid
        document_id = str(uuid.uuid4())
        
        # Upload PDF to S3
        logger.info(f"Uploading PDF to S3: {filename}")
        pdf_url = upload_pdf_to_s3(file_content, filename, document_id)
        logger.info(f"PDF uploaded to S3: {pdf_url}")
        
        # Process with Mistral OCR using the S3 URL
        logger.info(f"Processing PDF with Mistral OCR...")
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": pdf_url
            }
        )
        
        # Combine all pages into raw text
        raw_text_parts = []
        for page in ocr_response.pages:
            # Create raw text by removing markdown formatting
            raw_text = page.markdown.replace('#', '').replace('*', '').replace('_', '')
            raw_text_parts.append(raw_text)
        
        raw_text = "\n\n".join(raw_text_parts)
        
        # Save the extracted text as markdown in S3
        year = "misc"  # Default folder if we don't have a specific year
        md_filename = f"{document_id}.md"
        markdown_content = raw_text
        
        # Upload markdown content to S3
        from s3_utils import upload_markdown_to_s3
        upload_markdown_to_s3(markdown_content, year, md_filename)
        
        logger.info(f"Successfully extracted {len(raw_text)} characters with Mistral OCR")
        
        return raw_text
            
    except Exception as e:
        logger.error(f"Error processing with Mistral OCR: {str(e)}")
        raise Exception(f"Failed to process PDF with Mistral OCR: {str(e)}")