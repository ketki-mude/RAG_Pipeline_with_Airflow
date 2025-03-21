"""Module for extracting text from PDF documents using Docling"""
import io
import os
from pathlib import Path
from typing import Dict, Tuple, Any
from tempfile import NamedTemporaryFile
import logging

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Docling document converter
def get_document_converter():
    """Initialize and return Docling document converter"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    
    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    return doc_converter

# Global converter instance
_doc_converter = None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using Docling
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    global _doc_converter
    
    # Initialize converter if not already done
    if _doc_converter is None:
        _doc_converter = get_document_converter()
    
    try:
        # Convert the PDF file using Docling
        conv_result = _doc_converter.convert(pdf_path)
        
        # Extract raw text from the document
        raw_text = _extract_text_from_document(conv_result.document)
        
        # If we couldn't extract text, use the markdown content
        if not raw_text.strip():
            # Try to export markdown and strip formatting
            try:
                markdown_content = conv_result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
                raw_text = markdown_content.replace('#', '').replace('*', '')
            except Exception as e:
                logger.warning(f"Error exporting markdown: {str(e)}")
                # Try without specifying image_mode
                markdown_content = conv_result.document.export_to_markdown()
                raw_text = markdown_content.replace('#', '').replace('*', '')
        
        return raw_text
            
    except Exception as e:
        logger.error(f"Error processing PDF with Docling: {str(e)}")
        raise Exception(f"Failed to process PDF with Docling: {str(e)}")

def _extract_text_from_document(document) -> str:
    """
    Extract text from a Docling document
    
    Args:
        document: The Docling document
        
    Returns:
        The extracted text
    """
    raw_text = ""
    
    # Try different approaches to extract text
    try:
        # Try to get text directly from the document
        if hasattr(document, 'text'):
            logger.info("Using document.text")
            return document.text
        
        if hasattr(document, 'get_text_content'):
            logger.info("Using document.get_text_content()")
            return document.get_text_content()
        
        # Try to extract text from pages
        if hasattr(document, 'pages'):
            logger.info("Extracting text from pages")
            pages = document.pages
            
            # Try to iterate through pages
            try:
                for page in pages:
                    if hasattr(page, 'text'):
                        raw_text += page.text + "\n\n"
                    elif hasattr(page, 'get_text'):
                        raw_text += page.get_text() + "\n\n"
                    elif hasattr(page, 'blocks'):
                        # Extract text from blocks
                        for block in page.blocks:
                            if hasattr(block, 'text'):
                                raw_text += block.text + " "
                            elif hasattr(block, 'get_text'):
                                raw_text += block.get_text() + " "
                        raw_text += "\n\n"
            except Exception as e:
                logger.warning(f"Error iterating through pages: {str(e)}")
                # Try to access pages by index
                try:
                    for i in range(len(pages)):
                        page = pages[i]
                        if hasattr(page, 'text'):
                            raw_text += page.text + "\n\n"
                        elif hasattr(page, 'get_text'):
                            raw_text += page.get_text() + "\n\n"
                except Exception as e2:
                    logger.warning(f"Error accessing pages by index: {str(e2)}")
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
    
    return raw_text