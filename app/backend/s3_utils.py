import boto3
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# Add error checking for environment variables
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_S3_BUCKET_NAME]):
    raise ValueError("Missing required AWS credentials in .env file")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def test_s3_connection():
    """Test connection to S3 bucket"""
    try:
        s3_client.head_bucket(Bucket=AWS_S3_BUCKET_NAME)
        return True
    except Exception as e:
        print(f"S3 Connection Error: {str(e)}")
        return False

def ensure_s3_structure():
    """Ensure the basic folder structure exists in S3"""
    base_folders = [
        'documents/',
        'documents/pdf/',
        'documents/markdown/'
    ]
    
    try:
        for folder in base_folders:
            s3_client.put_object(
                Bucket=AWS_S3_BUCKET_NAME,
                Key=folder
            )
        return True
    except Exception as e:
        print(f"Error creating S3 structure: {e}")
        return False

def upload_file_to_s3(file_content: bytes, s3_key: str, content_type: str = None) -> str:
    """
    Uploads any file to S3.
    Returns URL for the uploaded file.
    """
    try:
        extra_args = {'ACL': 'public-read'}
        if content_type:
            extra_args['ContentType'] = content_type
            
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            **extra_args
        )
        return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        raise Exception(f"Failed to upload file to S3: {e}")
    
def upload_pdf_to_s3(file_content: bytes, original_filename: str, document_id: str) -> str:
    """
    Uploads PDF to S3.
    Returns URL for the uploaded file.
    """
    try:
        # Upload original PDF
        pdf_key = f"documents/pdf/{document_id}/{original_filename}"
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=pdf_key,
            Body=file_content,
            ContentType='application/pdf'
        )
        return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{pdf_key}"
    except Exception as e:
        raise Exception(f"Failed to upload PDF to S3: {e}")

def upload_markdown_to_s3(markdown_content: str, year: str, filename: str) -> str:
    """
    Uploads markdown content to S3.
    Returns URL for the uploaded file.
    
    Args:
        markdown_content: The markdown content to upload
        year: The year folder to use
        filename: The filename to use (should end with .md)
    """
    try:
        # Ensure consistent path structure with PDF storage
        markdown_key = f"documents/markdown/{year}/{filename}"
        
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=markdown_key,
            Body=markdown_content.encode('utf-8'),
            ContentType='text/markdown'
        )
        return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{markdown_key}"
    except Exception as e:
        raise Exception(f"Failed to upload markdown to S3: {e}")

def get_pdf_from_s3(document_id: str, filename: str) -> bytes:
    """
    Gets PDF content from S3.
    Returns the binary content of the PDF.
    """
    try:
        pdf_key = f"documents/pdf/{document_id}/{filename}"
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=pdf_key)
        return response['Body'].read()
    except Exception as e:
        raise Exception(f"Failed to get PDF from S3: {e}")

def get_markdown_from_s3(document_id: str, filename: str = None) -> str:
    """
    Gets markdown content from S3.
    Returns the markdown content as a string.
    
    Args:
        document_id: Document ID (format: YYYY_Quarter)
        filename: Optional filename (if different from document_id)
    """
    try:
        # Extract year from document_id
        year = document_id.split('_')[0]
        
        # Use filename if provided, otherwise use document_id
        md_filename = filename if filename else f"{document_id}.md"
        
        # Use consistent path structure
        markdown_key = f"documents/markdown/{year}/{md_filename}"
        
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=markdown_key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to get markdown from S3: {e}")

def list_documents_from_s3():
    """
    Lists all documents in S3 bucket.
    Returns a list of document IDs.
    """
    try:
        # List all objects in the PDF directory
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET_NAME,
            Prefix='documents/pdf/',
            Delimiter='/'
        )
        
        # Extract document IDs from CommonPrefixes
        document_ids = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                # Extract document ID from prefix
                prefix_path = prefix['Prefix']
                document_id = prefix_path.split('/')[-2]  # Format: documents/pdf/{document_id}/
                document_ids.append(document_id)
                
        return document_ids
    except Exception as e:
        raise Exception(f"Failed to list documents from S3: {e}")

def get_document_metadata(document_id: str):
    """
    Gets metadata for a document from S3.
    Returns a dictionary with document metadata.
    """
    try:
        # List all objects in the document's PDF directory
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET_NAME,
            Prefix=f'documents/pdf/{document_id}/'
        )
        
        if 'Contents' in response and len(response['Contents']) > 0:
            # Get the first (and should be only) PDF file
            pdf_object = response['Contents'][0]
            pdf_key = pdf_object['Key']
            filename = pdf_key.split('/')[-1]
            
            # Get object metadata
            head_response = s3_client.head_object(
                Bucket=AWS_S3_BUCKET_NAME,
                Key=pdf_key
            )
            
            # Extract last modified date
            last_modified = head_response['LastModified'].strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                'document_id': document_id,
                'original_filename': filename,
                'processing_date': last_modified,
                'pdf_url': f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{pdf_key}"
            }
        
        return None
    except Exception as e:
        raise Exception(f"Failed to get document metadata from S3: {e}")

# Run this when the module loads to ensure S3 structure exists
ensure_s3_structure()

if not test_s3_connection():
    print("WARNING: Cannot access S3 bucket. Please check your credentials and permissions.") 