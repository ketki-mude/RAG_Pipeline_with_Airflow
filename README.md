# RAG Pipeline with Airflow

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Apache Airflow. The system ingests NVIDIA quarterly reports (for the past 5 years), processes PDF documents using multiple parsing strategies, and integrates advanced retrieval methods with vector search for precise question answering.

---

## **📌 Project Resources**
- **Streamlit:** [Application Link](http://34.21.56.116:8501)
- **Airflow Dashboard:** [Airflow Link](http://34.21.56.116:8080)
- **Backend:** [Backend Link](http://34.21.56.116:8000)
- **Demo Video:** [YouTube Demo](https://youtu.be/7x4iwCADyJA)
- **ChromaDB:** [ChromaDBLink](http://34.21.56.116:8001)
- **Documentation:** [Codelab/Documentation Link](https://codelabs-preview.appspot.com/?file_id=1lXv5JZRfDRDjS80zOzsKx5Y2xpjeqESIqiHni75n_p8#4)


---

## **📌 Technologies Used**
<p align="center">
  <img src="https://img.shields.io/badge/-Apache_Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white" alt="Apache Airflow">
  <img src="https://img.shields.io/badge/-Docling-4B8BBE?style=for-the-badge" alt="Docling">
  <img src="https://img.shields.io/badge/-Mistral_OCR-FFCC00?style=for-the-badge" alt="Mistral OCR">
  <img src="https://img.shields.io/badge/-Pinecone-734BD4?style=for-the-badge" alt="Pinecone">
  <img src="https://img.shields.io/badge/-ChromaDB-34A853?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/-AWS_S3-569A31?style=for-the-badge&logo=amazon-s3&logoColor=white" alt="AWS S3">
</p>

---

## **📌 Architecture Diagram**
<p align="center">
  <img src="[https://your_link/architecture_diagram.jpg](https://github.com/ketki-mude/RAG_Pipeline_with_Airflow/blob/main/architecture 
       diagram/pdf_scraper_rag_based.png)" alt="Architecture Diagram" width="600">
</p>

---

## **📌 Project Flow**

### **Step 1: Data Collection & Ingestion**
- **Data Source:** NVIDIA quarterly reports for the past 5 years.
- **Workflow:** Automated data ingestion orchestrated by Apache Airflow.

### **Step 2: PDF Parsing**
- **Custom Extraction:** Extend previous extraction capabilities.
- **Docling:** Robust PDF parsing.
- **Mistral OCR:** Improved text extraction for scanned or complex layout documents.

### **Step 3: RAG Pipeline Construction**
- **Naive RAG Implementation:** Manual computation of embeddings and cosine similarity.
- **Vector Search Integrations:**
  - **Pinecone:** Scalable vector-based retrieval.
  - **ChromaDB:** Advanced document retrieval.
- **Chunking Strategies:** Employ multiple methods (fixed-size, semantic, and hybrid) to optimize retrieval and context extraction.

### **Step 4: Testing & User Interface**
- **Streamlit Application:** Allows users to:
  - Upload PDFs.
  - Select a PDF parser (Docling or Mistral OCR).
  - Choose a RAG method (manual embeddings, Pinecone, or ChromaDB).
  - Select the chunking strategy.
  - Filter by specific quarter data for targeted queries.
- **FastAPI Backend:** Connects the RAG pipeline with the UI.
- **LLM Integration:** Processes user queries and generates responses using a selected Large Language Model (LLM).

### **Step 5: Deployment**
- **Docker Pipelines:**
  - **Airflow Pipeline:** Manages data ingestion, processing, and retrieval workflows.
  - **Streamlit + FastAPI Pipeline:** Facilitates user interaction and query processing.

---

## **📌 Contributions**
| **Member**   | **Contribution**                                                                          |
|--------------|-------------------------------------------------------------------------------------------|
| **Janvi** | **33%** – Handled PDF parsing strategies (Docling and Mistral OCR) and FastAPI integration and pinecone vector search. |
| **Ketki** | **33%** – Responsible for data ingestion,  and integrating Chromadb for vector search. |
| **Sahil** | **33%** – Focused on RAG implementation, Manual embeddings integration, and developing the Streamlit UI and Airflow orchestration. |

---

## **📌 Attestation**
**WE CERTIFY THAT WE HAVE NOT USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND COMPLY WITH THE POLICIES OUTLINED IN THE STUDENT HANDBOOK.**
