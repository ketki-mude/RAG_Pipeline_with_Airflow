import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from app.backend.sentenceChunking import get_text_chunks
from dotenv import load_dotenv

class PineconeResponder:
    def __init__(self, index_name="pdf-scrapper-index", dimension=384, top_k=7):
        """Initialize Pinecone, Gemini API, and the embedding model."""
        # Load environment variables
        dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        load_dotenv(dotenv_path)
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.dimension = dimension
        self.top_k = top_k
        
        # Configure Gemini API
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Create the index if it does not exist
        if self.index_name not in [index["name"] for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' already exists. Skipping creation.")
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        
        # Load the embedding model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def ingest_file(self, file_path, pdf_id):
        """
        Read a file, split it into chunks, generate embeddings, and store them in Pinecone
        with metadata indicating the source PDF.
        Parameters:
            file_path (str): Path to the PDF file converted to text (or a Markdown file).
            pdf_id (str): An identifier for the PDF (could be a filename or unique ID).
        """
        chunks = get_text_chunks(file_path)
        embeddings = self.model.encode(chunks).tolist()
        
        # Insert each chunk into Pinecone with metadata that includes the pdf_id
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.index.upsert([
                {
                    "id": f"{pdf_id}_chunk_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "pdf_id": pdf_id  # Adding metadata to identify the PDF source
                    }
                }
            ])
        
        print(f"Chunks from {pdf_id} inserted into Pinecone successfully!")


    def _get_namespace(self, file_path):
        """Generate a namespace from the file name (without extension)."""
        return os.path.basename(file_path).split('.')[0].replace(" ", "_")


    def _count_tokens(self, text):
        """
        Count the number of tokens in a text by splitting on whitespace.
        Parameters:
            text (str): The text to count tokens for.
        Returns:
            int: The approximate number of tokens.
        """
        return len(text.split())

    def search(self, query, pdf_id_filter=None):
        """
        Search for relevant chunks in Pinecone, filtering by a specific pdf_id, and generate response using Gemini.
        Parameters- query (str): The user's query.
                    pdf_id_filter (str, optional): If provided, restricts the search to chunks from this PDF.
        Return- str: The generated answer from Gemini.
        """
        query_embedding = self.model.encode([query]).tolist()
        
        # Build a filter if pdf_id_filter is provided
        filter_params = {"pdf_id": pdf_id_filter} if pdf_id_filter else None
        
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True,
            filter=filter_params
        )
        
        # Extract relevant chunks
        retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]
        
        # Create a context for Gemini
        context = "\n".join(retrieved_texts)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"

        # Count tokens for the prompt (tokens sent to Gemini)
        tokens_sent = self._count_tokens(prompt)
        
        # Generate response using Gemini
        response = self.gemini_model.generate_content(prompt)
        answer_text = response.text

        # Count tokens for the response (tokens received from Gemini)
        tokens_received = self._count_tokens(answer_text)

        # Print token count details
        print("=== Token Details ===")
        print("Tokens sent to Gemini:", tokens_sent)
        print("Tokens received from Gemini:", tokens_received)
        
        return answer_text
    
    def summarize_data(self, pdf_id_filter=None, token_limit=500):
        """
        Summarize the ingested data (optionally filtered by pdf_id) with a fixed token limit.
        Parameters: pdf_id_filter (str, optional): If provided, restricts the summary to chunks from that PDF.
                    token_limit (int): Maximum number of tokens for the summary.
        Returns: str: The generated summary.
        """
        # Build a filter if pdf_id_filter is provided
        filter_params = {"pdf_id": pdf_id_filter} if pdf_id_filter else None

        # Set a high top_k to retrieve all chunks for the document.
        # Note: If the document is very long, you might need to adopt a hierarchical summarization strategy.
        results = self.index.query(
            vector=self.model.encode(["summarize content"]).tolist()[0],  # Using a dummy query embedding
            top_k=1500,  # Use a high value to retrieve all chunks; adjust as necessary
            include_metadata=True,
            filter=filter_params
        )
        
        # Extract text from the retrieved chunks
        retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]
        context = "\n".join(retrieved_texts)
        
        # Build a robust prompt that instructs the LLM clearly
        prompt = (
            f"You are an expert summarizer. Please provide a concise and coherent summary of content. "
            f"Focus on the main ideas and important details. "
            f"Your summary must not exceed {token_limit} tokens.\n"
            f"Document Content: {context}"
        )
        
        # Generate summary using Gemini
        summary_response = self.gemini_model.generate_content(prompt)
        summary_text = summary_response.text
        
        # Optionally, count the tokens in the summary
        print("Summary input token count:", self._count_tokens(context))
        print("Summary output token count:", self._count_tokens(summary_text))
        
        return summary_text


if __name__ == "__main__":
    responder = PineconeResponder()
    
    # Example: Get a namespace from a file path
    file_path = "/Users/janvichitroda/Documents/Janvi/NEU/Big_Data_Intelligence_Analytics/Assignment 4/Part 2/LLM_With_Pinecone/PineconeHandson/InputFiles/inputFile.md"
    pdf_id_filter = responder._get_namespace(file_path)
    print("Namespace:", pdf_id_filter)
    
    # Ingest a file with a PDF identifier derived from its filename
    responder.ingest_file(file_path, pdf_id=pdf_id_filter)
    
    # Query across all PDFs
    query_text = "What are the key factors supporting corporate earnings growth in 2025?"
    response_all = responder.search(query_text)
    print("Response (All PDFs):\n", response_all)
    
    # Query a specific PDF by filtering with pdf_id
    response_filtered = responder.search(query_text, pdf_id_filter=pdf_id_filter)
    print("Response (Filtered by PDF):\n", response_filtered)
    
    # Summarize data with a fixed token limit (optionally for a specific PDF)
    summary = responder.summarize_data(pdf_id_filter=pdf_id_filter, token_limit=100)
    print("Summary:\n", summary)