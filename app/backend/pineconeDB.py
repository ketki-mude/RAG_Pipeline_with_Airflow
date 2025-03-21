from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentenceChunking import get_text_chunks

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-scrapper-index"
dimension = 384  # Dimension for all-MiniLM-L6-v2 model

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Check if the index exists before creating it
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust as needed
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")


# Connect to the index
index = pc.Index(index_name)

# Load the embedding model from Hugging Face
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get text chunks from the input file
file_path = "/Users/janvichitroda/Documents/Janvi/NEU/Big_Data_Intelligence_Analytics/Assignment 4/Part 2/LLM_With_Pinecone/PineconeHandson/InputFiles/inputFile.md"
chunks = get_text_chunks(file_path)

# Generate embeddings for chunks
embeddings = model.encode(chunks).tolist()

# Insert data into Pinecone
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    index.upsert([
        {"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk}}
    ])

print("Chunks inserted into Pinecone successfully!")

# def search_pinecone(query, top_k=7):
#     """Search for the most relevant chunks in Pinecone."""
#     query_embedding = model.encode([query]).tolist()
#     results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
#     print(f"Top {top_k} results for query '{query}':\n",results)
#     for match in results["matches"]:
#         print(f"Score: {match['score']:.4f} | Chunk: {match['metadata']['text']}\n")

# # Example search query
# query_text = "What is one key reason behind the expected increase in Mergers & Acquisitions (M&A) in 2025?"
# search_pinecone(query_text)

def search_pinecone(query, top_k=7):
    """Search for relevant chunks in Pinecone and generate a response using Gemini."""
    query_embedding = model.encode([query]).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    print(results)

    # Extract relevant chunks
    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]

    # Create a context for Gemini
    context = "\n".join(retrieved_texts)
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"

    # Generate response using Gemini
    response = gemini_model.generate_content(prompt)
    
    return response.text

# Example search query
query_text = "What are the key factors supporting corporate earnings growth in 2025?"
response_text = search_pinecone(query_text)
print("Gemini Response:\n", response_text)