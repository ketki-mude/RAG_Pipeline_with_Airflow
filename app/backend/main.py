from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid
import time
from datetime import datetime
from contextlib import asynccontextmanager
from chromadb import PersistentClient

# Import existing modules
from pdf_extractor import extract_text_from_pdf  # For Docling
from mistral_ocr_extractor import extract_text_with_mistral, process_uploaded_pdf_with_mistral  # For Mistral OCR
from chunking_strategies import chunk_document  # The main chunking function
from embedding_service import generate_embeddings  # For embeddings generation
from vector_storage_service import (
    store_in_chromadb,
    search_chromadb,
    store_in_pinecone,
    search_pinecone
)
from llm_service import generate_response_with_gemini  # Use only Gemini as per requirement

# Pass the lifespan to FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "text")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
EMBEDDINGS_FILE = "data/embeddings.json"

# Configuration for vector DBs
CHROMA_HOST = "chroma-db"  # Docker service name in Airflow
CHROMA_PORT = 8000
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

# Ensure directories exist
for directory in [PDF_DIR, TEXT_DIR, CHUNKS_DIR, EMBEDDINGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    vector_db: str
    quarter_filter: Optional[List[str]] = None
    top_k: int = 5
    document_id: Optional[str] = None  # For user-uploaded PDFs

class UploadPdfRequest(BaseModel):
    extractor: str = "docling"  # "docling" or "mistral_ocr"
    chunking_method: str = "fixed"  # "fixed", "markdown", or "sentence"
    vector_db: str = "chromadb"  # "chromadb", "pinecone", or "embeddings"
    quarter: Optional[str] = None

class AvailableQuartersResponse(BaseModel):
    quarters: List[str]

# API Endpoints
@app.get("/")
async def root():
    return {"message": "NVIDIA Financial RAG API is running"}

@app.get("/health")
async def health_check():
    """Check the health of connected services"""
    status = {
        "api": "healthy",
        "chromadb": "unknown",
        "pinecone": "unknown",
        "embeddings_file": "unknown",
    }
    
    # Check ChromaDB
    if hasattr(app, 'chroma_client'):
        try:
            collections = app.chroma_client.list_collections()
            status["chromadb"] = "healthy"
            status["collections"] = [col.name for col in collections]
        except Exception as e:
            status["chromadb"] = f"unhealthy: {str(e)}"
    
    # Check Pinecone
    if PINECONE_API_KEY:
        try:
            import pinecone
            indexes = pinecone.list_indexes()
            status["pinecone"] = "healthy"
            status["indexes"] = indexes
        except Exception as e:
            status["pinecone"] = f"unhealthy: {str(e)}"
    else:
        status["pinecone"] = "not configured"
    
    # Check embeddings file
    if os.path.exists(EMBEDDINGS_FILE):
        status["embeddings_file"] = "healthy"
        try:
            with open(EMBEDDINGS_FILE, 'r') as f:
                data = json.load(f)
            status["embeddings_count"] = len(data)
        except Exception as e:
            status["embeddings_file"] = f"exists but invalid: {str(e)}"
    else:
        status["embeddings_file"] = "not found"

    return status

@app.get("/available_quarters", response_model=AvailableQuartersResponse)
async def get_available_quarters():
    """Get all available quarters from the vector databases"""
    quarters = set()
    
    # Check ChromaDB
    if hasattr(app, 'chroma_client'):
        try:
            collection = app.chroma_client.get_collection(name="nvidia_financials")
            # Query to get metadata with quarters
            sample_results = collection.query(
                query_texts=["nvidia revenues"],
                n_results=100
            )
            
            for metadata_list in sample_results.get("metadatas", []):
                if metadata_list:
                    for metadata in metadata_list:
                        if "quarter" in metadata:
                            quarters.add(metadata["quarter"])
        except Exception as e:
            print(f"Error fetching quarters from ChromaDB: {str(e)}")
    
    # Check embeddings.json
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'r') as f:
                data = json.load(f)
            
            for item in data:
                if "metadata" in item and "quarter" in item["metadata"]:
                    quarters.add(item["metadata"]["quarter"])
        except Exception as e:
            print(f"Error fetching quarters from embeddings.json: {str(e)}")
    
    # If we found no quarters, provide some defaults
    if not quarters:
        quarters = {"2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
                    "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
                    "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
                    "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
                    "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"}
    
    return {"quarters": sorted(list(quarters))}

@app.post("/process_pdf")
async def process_pdf(
    file: UploadFile = File(...),
    extractor: str = Form("docling"),
    chunking_method: str = Form("fixed"),
    vector_db: str = Form("chromadb"),
    quarter: str = Form(None)
):
    """Process a PDF file with the specified options"""
    try:
        # Generate a unique document ID
        document_id = str(uuid.uuid4())
        filename = f"{document_id}_{file.filename}"
        
        # Read the file content
        file_content = await file.read()
        
        # Extract text using the selected extractor
        if extractor.lower() == "mistral_ocr":
            # Use the new direct upload function for Mistral OCR
            # This will upload to S3 and use the S3 URL
            text_content = process_uploaded_pdf_with_mistral(
                file_content=file_content,
                filename=filename  # Use the full filename with document_id prefix
            )
        else:  # Default to docling
            # For docling, we still need to save the file to disk
            pdf_path = os.path.join(PDF_DIR, filename)
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(file_content)
            text_content = extract_text_from_pdf(pdf_path)
        
        # Create chunks using the selected method
        chunks = chunk_document(
            text_content, 
            strategy=chunking_method, 
            chunk_size=500,  # Default chunk size for fixed method
            chunk_overlap=50  # Default overlap
        )
        
        # Create metadata for the chunks
        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "processing_date": datetime.now().isoformat(),
            "extractor": extractor,
            "chunking_method": chunking_method
        }
        
        # Add quarter if provided
        if quarter:
            metadata["quarter"] = quarter
            
        # Generate embeddings for chunks using sentence-transformers/all-MiniLM-L6-v2
        embeddings_data = []
        
        for i, chunk in enumerate(chunks):
            # Add chunk-specific metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            
            # Generate embedding using sentence-transformers
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
        
        # Store in the selected vector database
        if vector_db.lower() == "chromadb":
            collection_name = "user_uploads"
            
            # Use the app's persistent ChromaDB client instead of creating a new one
            if hasattr(app, 'chroma_client') and app.chroma_client:
                client = app.chroma_client
                
                # Get or create the collection
                collection = client.get_or_create_collection(name=collection_name)
                
                # Extract the necessary data for ChromaDB
                documents = []
                metadatas = []
                embeddings_list = []
                ids = []
                
                for i, item in enumerate(embeddings_data):
                    documents.append(item.get("content", ""))
                    metadatas.append(item.get("metadata", {}))
                    embeddings_list.append(item.get("embedding", []))
                    # Generate a unique ID for each document
                    ids.append(f"{collection_name}_{document_id}_{i}")
                
                # Add the documents to the collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings_list,
                    ids=ids
                )
                
                print(f"Successfully stored {len(embeddings_data)} documents in ChromaDB collection '{collection_name}'")
            else:
                # Fall back to the original function if app client isn't available
                store_in_chromadb(embeddings_data, collection_name=collection_name)
        
        elif vector_db.lower() == "pinecone":
            index_name = "user-uploads"
            store_in_pinecone(embeddings_data, index_name=index_name)
        
        else:  # embeddings.json
            # For embeddings.json, we'll create a user-specific file
            user_embeddings_file = os.path.join(EMBEDDINGS_DIR, f"{document_id}_embeddings.json")
            with open(user_embeddings_file, 'w') as f:
                json.dump(embeddings_data, f)
        
        return {
            "message": "PDF processed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "vector_db": vector_db
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer a question using RAG with the specified vector DB and quarter filter"""
    try:
        start_time = time.time()
    
        # Determine which vector DB to query
        vector_db = request.vector_db.lower()
        
        # Get context chunks
        if vector_db == "chromadb":
            # Query the collection based on document_id or quarterly data
            collection_name = "user_uploads" if request.document_id else "nvidia_financials"
            
            # Prepare filter
            where_filter = {}
            if request.quarter_filter and len(request.quarter_filter) > 0:
                where_filter["quarter"] = {"$in": request.quarter_filter}
            if request.document_id:
                where_filter["document_id"] = request.document_id
                
            # Search ChromaDB    
            context_chunks = search_chromadb(
                request.question,
                collection_name=collection_name,
                where_filter=where_filter if where_filter else None,
                top_k=request.top_k
            )
            
        elif vector_db == "pinecone":
            print("Searching Pinecone")
            # Prepare filter
            filter_dict = {}
            if request.quarter_filter and len(request.quarter_filter) > 0:
                filter_dict["quarter"] = {"$in": request.quarter_filter}
            if request.document_id:
                filter_dict["document_id"] = request.document_id
                
            # Search Pinecone
            index_name = "user-uploads" if request.document_id else "nvidia-financials"
            context_chunks = search_pinecone(
                request.question,
                index_name=index_name,
                filter_dict=filter_dict if filter_dict else None,
                top_k=request.top_k
            )
            
        else:  # embeddings.json
            # Determine which embeddings file to use
            embeddings_file = EMBEDDINGS_FILE
            print(f"Using embeddings file: {embeddings_file}")
            print("Request: ", request)
            if request.document_id:
                user_embeddings_file = os.path.join(EMBEDDINGS_DIR, f"embeddings.json")
                if os.path.exists(user_embeddings_file):
                    embeddings_file = user_embeddings_file
            
            # Search embeddings file
            context_chunks = search_embeddings_json(
                request.question,
                embeddings_file=embeddings_file,
                quarter_filter=request.quarter_filter,
                top_k=request.top_k
            )
        
        # If no context chunks found, return an appropriate message
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try a different question or adjust your filters.",
                "context_chunks": [],
                "processing_time": time.time() - start_time,
                "token_info": None
            }
        
        # Extract text from context chunks
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        # Generate answer with Gemini
        answer, token_info = generate_response_with_gemini(
            request.question,
            context_text,
            model_name="gemini-1.5-pro"  # Using Gemini as per your requirement
        )
        
        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "processing_time": time.time() - start_time,
            "token_info": token_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Function to search embeddings.json
def search_embeddings_json(question, embeddings_file, quarter_filter=None, top_k=5):
    """Search embeddings.json file for relevant chunks"""
    if not os.path.exists(embeddings_file):
        raise HTTPException(status_code=404, detail=f"Embeddings file not found: {embeddings_file}")
    
    try:
        # Load embeddings
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        # Handle different possible JSON structures
        # Check if data is a list or dictionary
        if isinstance(data, dict) and "documents" in data:
            # Handle VectorStorageService format (from class implementation)
            embeddings_list = []
            for doc_id, doc_data in data["documents"].items():
                if "chunks" in doc_data:
                    for chunk in doc_data["chunks"]:
                        embeddings_list.append({
                            "content": chunk.get("text", ""),
                            "embedding": chunk.get("embedding", []),
                            "metadata": {
                                "document_id": doc_id,
                                "quarter": doc_data.get("metadata", {}).get("quarter", "unknown"),
                                **doc_data.get("metadata", {})
                            }
                        })
            data = embeddings_list
        elif not isinstance(data, list):
            # Convert to list if it's not already
            raise ValueError("Unsupported embeddings file format. Expected a list or a dictionary with 'documents' key.")
        
        # Filter by quarters if provided
        if quarter_filter and len(quarter_filter) > 0:
            filtered_data = []
            for item in data:
                # Check if metadata exists and has quarter info
                metadata = item.get("metadata", {})
                item_quarter = None
                
                # Try different ways quarter might be stored
                if isinstance(metadata, dict) and "quarter" in metadata:
                    item_quarter = metadata["quarter"]
                    
                # Only include items with matching quarters
                if item_quarter and item_quarter in quarter_filter:
                    filtered_data.append(item)
                
            filtered_data = data
        else:
            filtered_data = data
        
        # If no data after filtering, return empty results
        if not filtered_data:
            return []
        
        # Generate embedding for the question
        question_embedding = generate_embeddings(
            question,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Compute cosine similarity with all chunks
        import numpy as np
        
        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        similarities = []
        for item in filtered_data:
            embedding_key = "embedding"
            content_key = "content"
            
            # Handle different possible keys
            if embedding_key not in item:
                if "vector" in item:
                    embedding_key = "vector"
                elif "embeddings" in item:
                    embedding_key = "embeddings"
                    
            if content_key not in item:
                if "text" in item:
                    content_key = "text"
                elif "chunk" in item:
                    content_key = "chunk"
            
            # Skip items without embeddings or content
            if embedding_key not in item or content_key not in item:
                continue
                
            # Get the embedding
            item_embedding = item[embedding_key]
            
            # Calculate similarity
            try:
                similarity = cosine_similarity(
                    np.array(question_embedding), 
                    np.array(item_embedding)
                )
                
                # Get metadata
                metadata = item.get("metadata", {})
                
                # Add to results
                similarities.append({
                    "text": item[content_key],
                    "document_id": metadata.get("document_id", "unknown") if isinstance(metadata, dict) else "unknown",
                    "filename": metadata.get("filename", "unknown") if isinstance(metadata, dict) else "unknown",
                    "similarity": float(similarity)
                })
            except Exception as e:
                print(f"Error calculating similarity for item: {e}")
                continue
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        # Return top-k results
        print("Similarities sorted: ", similarities[:top_k])
        return similarities[:top_k]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error searching embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)