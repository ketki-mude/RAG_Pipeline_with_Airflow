FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    wget \
    git \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY app/backend/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY app/backend /app/backend
COPY app/shared /app/shared

# Create necessary directories
RUN mkdir -p /app/data/embeddings \
    && mkdir -p /app/data/chroma_db \
    && mkdir -p /app/data/pdfs \
    && mkdir -p /app/data/uploads

# Download NLTK data
RUN python -m nltk.downloader punkt 

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2"

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]