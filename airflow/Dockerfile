FROM apache/airflow:2.6.3

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install Python dependencies
COPY airflow/requirements-airflow.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Create necessary directories
RUN mkdir -p /opt/airflow/data/embeddings \
    && mkdir -p /opt/airflow/data/chroma_db \
    && mkdir -p /opt/airflow/data/pdfs

# Set work directory
WORKDIR /opt/airflow