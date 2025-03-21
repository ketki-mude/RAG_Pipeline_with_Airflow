from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import your pipeline modules - this assumes you have your app code accessible in PYTHONPATH
# You may need to add your app directory to the PYTHONPATH in the Dockerfile
import sys
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/app')

# Import your pipeline functions
from app.backend.nvidia_pipeline import (
    fetch_pdf_s3_upload,
    convert_markdown_s3_upload,
    process_chunks_and_embeddings,
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'nvidia_financial_reports_pipeline',
    default_args=default_args,
    description='Process NVIDIA financial reports and create embeddings',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['nvidia', 'rag', 'embeddings'],
)

# Define the tasks
fetch_pdfs_task = PythonOperator(
    task_id='fetch_pdfs',
    python_callable=fetch_pdf_s3_upload,
    dag=dag,
)

convert_to_markdown_task = PythonOperator(
    task_id='convert_to_markdown',
    python_callable=convert_markdown_s3_upload,
    op_kwargs={'reports': "{{ task_instance.xcom_pull(task_ids='fetch_pdfs') }}"},
    dag=dag,
)

process_embeddings_task = PythonOperator(
    task_id='process_embeddings',
    python_callable=process_chunks_and_embeddings,
    op_kwargs={
        'processed_reports': "{{ task_instance.xcom_pull(task_ids='convert_to_markdown') }}",
        'chunking_strategy': 'markdown',
    },
    dag=dag,
)

# Set task dependencies
fetch_pdfs_task >> convert_to_markdown_task >> process_embeddings_task