#!/bin/bash
set -e

# Create necessary directories
mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins /opt/airflow/data
mkdir -p /opt/airflow/data/embeddings /opt/airflow/data/chroma_db /opt/airflow/data/pdfs

# Initialize Airflow database
# Note the --skip-encryption flag for development
airflow db init --skip-encryption

# Create Airflow admin user
airflow users create \
    --username airflow \
    --password airflow \
    --firstname John \
    --lastname Doe \
    --role Admin \
    --email admin@example.com

# Set correct permissions
chown -R "50000:0" /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins /opt/airflow/data

echo "Airflow initialization complete"
