from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.onprem.client import Users
from diagrams.gcp.compute import Run
from diagrams.aws.storage import S3
from diagrams.custom import Custom

# Set diagram formatting
graph_attr = {
    "fontsize": "24",
    "bgcolor": "white",
    "splines": "ortho",
}

# Base path for images (Updated to your absolute path)
base_path = r"input_icons"

# Create the diagram
with Diagram("pdf_scraper_rag_based", show=False, graph_attr=graph_attr, direction="TB"):
   
    # User/Client
    user = Users("End User")
   
    # Frontend Cluster
    with Cluster("Frontend (User Interface)"):
        streamlit = Custom("Streamlit UI", f"{base_path}/streamlit.png")

    # Select a PDF
    with Cluster("PDF Cloud Storage"):
        pdf_upload = Custom("Select existing PDF \n Upload new PDF", f"{base_path}/s3_image.png")
   
    # Cloud Infrastructure Cluster
    with Cluster("GCP VM Instance"):
        # GCP Cloud Run hosting the FastAPI backend
        cloud_run = Custom("GCP VM Instance", f"{base_path}/gcp.png")

        with Cluster("Backend"):
            fastapi = Custom("FastAPI", f"{base_path}/FastAPI.png")
        
        with Cluster("Backend"):
            airflow = Custom("Airflow", f"{base_path}/Airflow.png")

        with Cluster("RAG Storage"):
            rag = Custom("Summary and QnA \n Request Stream", f"{base_path}/rag.png")

            with Cluster("RAG Storage"):
                manualEmbed = Custom("Manual Vector Embedding", f"{base_path}/manual_embed.png")
                pinecone = Custom("Pinecone Vector DB", f"{base_path}/pinecone.png")
                chroma = Custom("Chroma Vector DB", f"{base_path}/chroma.png")

            with Cluster("LLM Integration"):
                llm = Custom("LLM Integration", f"{base_path}/llm.png")
                zephyr = Custom("Zephyr \n (HuggingFace)", f"{base_path}/huggingface.png")
                gemini = Custom("Gemini \n (Google)", f"{base_path}/gemini.png")

    user >> streamlit
    streamlit >> user
    streamlit >> pdf_upload 
    
    streamlit >> cloud_run
    cloud_run >> airflow >> rag
    cloud_run >> fastapi >> rag

    pdf_upload >> streamlit
    cloud_run >> streamlit

    pdf_upload >> cloud_run
    pdf_upload >> cloud_run

    fastapi >> cloud_run
    airflow >> cloud_run
    rag >> fastapi
    rag >> airflow

    with Cluster("Processing Flows"):     
        rag >> manualEmbed
        manualEmbed >> rag
        manualEmbed >> llm

        rag >> pinecone
        pinecone >> rag
        pinecone >> llm 

        rag >> chroma
        chroma >> rag
        chroma >> llm

        llm >> [zephyr, gemini]
        [zephyr, gemini] >> llm

