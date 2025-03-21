from webscraper import fetch_nvidia_financial_reports
from mistral_ocr_extractor import process_uploaded_pdf_with_mistral
from chunking_strategies import chunk_document
from embedding_service import generate_embeddings
from vector_storage_service import store_in_chromadb, store_in_pinecone, store_embeddings_json
from s3_utils import get_markdown_from_s3, upload_markdown_to_s3
import time
import os
import uuid

def fetch_pdf_s3_upload():
    """Fetch NVIDIA financial reports and upload to S3"""
    # Step 1: Fetch NVIDIA financial reports
    print("Step 1: Fetching financial reports...")
    reports = fetch_nvidia_financial_reports()
    print("Reports fetched successfully:")
    for report in reports:
        print(f"Fetched: {report['pdf_filename']} (Size: {len(report['content'])} bytes)")
    return reports

def convert_markdown_s3_upload(reports):
    """Convert PDFs to markdown using Mistral OCR and upload to S3"""
    processed_reports = []
    
    for report in reports:
        pdf_filename = report["pdf_filename"]
        pdf_content = report["content"]
        s3_url = report.get("s3_url", "")
        
        # Use a naming convention: use the part before the dot as document_id
        document_id = pdf_filename.split('.')[0]
        
        try:
            # Process the PDF using the new function-based approach
            print(f"Processing {pdf_filename} with Mistral OCR...")
            
            # Extract text using Mistral OCR - this now also uploads to S3 internally
            markdown_content = process_uploaded_pdf_with_mistral(
                file_content=pdf_content,
                filename=pdf_filename
            )
            
            print(f"Extracted markdown content: {len(markdown_content)} characters")
            
            # Add the processed report to our list
            processed_reports.append({
                "document_id": document_id,
                "pdf_filename": pdf_filename,
                "original_filename": pdf_filename,
                "content_length": len(markdown_content),
                "markdown_content": markdown_content,  # Store content for chunking
                "s3_url": s3_url
            })
            
        except Exception as e:
            print(f"Error converting {pdf_filename} to markdown: {e}")
    
    return processed_reports

def process_chunks_and_embeddings(processed_reports, chunking_strategy="markdown"):
    """
    Process all reports by chunking them and storing embeddings in various vector stores
    
    Args:
        processed_reports: List of processed report details
        chunking_strategy: Which chunking strategy to use
    """
    print(f"\nStep 3: Chunking documents using {chunking_strategy} strategy and creating embeddings...")
    
    for report in processed_reports:
        document_id = report["document_id"]
        markdown_content = report.get("markdown_content", "")
        
        try:
            # Extract year from document_id
            parts = document_id.split('_')
            year = parts[0]
            
            # Extract quarter information
            quarter = None
            quarter_mapping = {
                "First": "Q1",
                "Second": "Q2", 
                "Third": "Q3",
                "Fourth": "Q4",
                "Q1": "Q1",
                "Q2": "Q2",
                "Q3": "Q3", 
                "Q4": "Q4"
            }
            
            # Check for quarter information in document_id
            for part in parts:
                if part in quarter_mapping:
                    quarter = quarter_mapping[part]
                    break
                # Check for "Quarter" in parts (like "Fourth_Quarter")
                elif len(parts) > 1 and "Quarter" in parts:
                    idx = parts.index("Quarter")
                    if idx > 0 and parts[idx-1] in quarter_mapping:
                        quarter = quarter_mapping[parts[idx-1]]
                        break
            
            # If still no quarter, try to find quarter-related terms
            if not quarter:
                for i, part in enumerate(parts):
                    if "Quarter" in part:
                        # Check if previous part indicates which quarter
                        if i > 0 and parts[i-1] in quarter_mapping:
                            quarter = quarter_mapping[parts[i-1]]
                            break
            
            # Default to "Unknown" if we couldn't extract the quarter
            if not quarter:
                print(f"Warning: Could not extract quarter from document_id: {document_id}")
                quarter = "Unknown"
            else:
                print(f"Extracted quarter: {quarter} from document_id: {document_id}")
            
            # If we don't have the markdown content in memory (from a previous step)
            if not markdown_content:
                try:
                    # Get markdown content from S3 using the correct path
                    print(f"Retrieving markdown for {document_id} from S3...")
                    markdown_content = get_markdown_from_s3(document_id)
                except Exception as e:
                    print(f"Error retrieving markdown from S3: {e}")
                    continue
            
            # Create chunks using specified strategy
            print(f"Chunking document using {chunking_strategy} strategy...")
            chunks = chunk_document(markdown_content, strategy=chunking_strategy)
            print(f"Created {len(chunks)} chunks")
            
            # Create standardized time period format
            time_period = f"{year}-{quarter}"
            
            # Document metadata
            metadata = {
                "document_id": document_id,
                "source_type": "pdf",
                "original_filename": report["pdf_filename"],
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunking_strategy": chunking_strategy,
                "url": report.get("s3_url", ""),
                "year": year,
                "quarter": quarter,
                "time_period": time_period  # Adding standardized time period
            }
            
            # Generate embeddings and prepare data for storage
            print(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings_data = []
            
            for i, chunk in enumerate(chunks):
                # Add chunk-specific metadata
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                
                # Generate embedding
                embedding = generate_embeddings(
                    chunk, 
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Add to embeddings data
                embeddings_data.append({
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": chunk_metadata
                })
            
            # Store in all vector databases
            print(f"Storing {len(embeddings_data)} chunks in ChromaDB...")
            store_in_chromadb(embeddings_data, collection_name="nvidia_financials")
            
            print(f"Storing {len(embeddings_data)} chunks in Pinecone...")
            store_in_pinecone(embeddings_data, index_name="nvidia-financials")
            
            print(f"Storing {len(embeddings_data)} chunks in embeddings.json...")
            store_embeddings_json(embeddings_data, file_path="data/embeddings/nvidia_embeddings.json")
            
            print(f"Successfully processed document {document_id}")
            
        except Exception as e:
            print(f"Error processing document {document_id}: {e}")

def run_pipeline(chunking_strategy="markdown"):
    """
    Run the complete NVIDIA pipeline with specified chunking strategy
    
    Args:
        chunking_strategy: Chunking strategy to use (markdown, sentence, or fixed)
    """
    print(f"Starting NVIDIA financial reports pipeline with {chunking_strategy} chunking...")
    
    # Step 1: Fetch PDFs and upload to S3
    reports = fetch_pdf_s3_upload()
    
    # Step 2: Convert PDFs to markdown and upload to S3
    processed_reports = convert_markdown_s3_upload(reports)
    
    # Step 3: Process chunks and create embeddings
    process_chunks_and_embeddings(processed_reports, chunking_strategy)
    
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    # Set up necessary directories
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs("data/chroma_db", exist_ok=True)
    
    # Run the complete pipeline with markdown chunking strategy
    run_pipeline(chunking_strategy="markdown")
    
    # Uncomment to run pipeline with other chunking strategies
    # run_pipeline(chunking_strategy="sentence")
    # run_pipeline(chunking_strategy="fixed")