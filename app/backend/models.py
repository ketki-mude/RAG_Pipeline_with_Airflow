from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    processing_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    content: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    processing_date: str
    pdf_url: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]

class DocumentContentResponse(BaseModel):
    document_id: str
    original_filename: str
    content: str
    markdown_content: str
    metadata: Optional[Dict[str, Any]] = None

class SummarizeRequest(BaseModel):
    document_id: str
    model_id: str = "huggingface/HuggingFaceH4/zephyr-7b-beta"

class SummarizeResponse(BaseModel):
    summary: str
    cost: Dict[str, Any]

class QuestionRequest(BaseModel):
    document_id: str
    question: str
    model_id: str = "huggingface/HuggingFaceH4/zephyr-7b-beta"

class QuestionResponse(BaseModel):
    answer: str
    cost: Dict[str, Any]

class QuestionRAGRequest(BaseModel):
    question: str
    model_id: str = "huggingface/HuggingFaceH4/zephyr-7b-beta"

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    similarity: float
    document_metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult] 