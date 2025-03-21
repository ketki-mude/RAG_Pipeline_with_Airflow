import re
import nltk
from typing import List, Optional

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # If punkt_tab isn't available as a separate download,
        # we'll need to handle that in the sentence_chunks function
        print("Note: punkt_tab not available as a direct download. Will use punkt instead.")

from nltk.tokenize import sent_tokenize

def markdown_header_chunks(text: str) -> List[str]:
        """
        Chunk text based on markdown headers.
        
        Args:
            text: The markdown text to chunk.
            
        Returns:
            List of text chunks with headers as separation points.
        """
        # Regular expression to find markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
        
        # Find all headers and their positions
        headers = [(match.start(), match.group()) for match in header_pattern.finditer(text)]
        
        # If no headers found, return whole text as one chunk
        if not headers:
            return [text.strip()]
        
        chunks = []
        
        # First chunk includes everything before the first header
        if headers[0][0] > 0:
            chunks.append(text[:headers[0][0]].strip())
        
        # Process chunks between headers
        for i in range(len(headers)):
            start_pos = headers[i][0]
            # If this is the last header, end position is the end of text
            if i == len(headers) - 1:
                end_pos = len(text)
            else:
                end_pos = headers[i+1][0]
            
            # Add the chunk, including the header
            chunk = text[start_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
        
        return [chunk for chunk in chunks if chunk]  # Remove any empty chunks
    
def sentence_chunks(text: str) -> List[str]:
        """
        Split text into chunks based on sentences.
        
        Args:
            text: The text to split into sentence-based chunks.
            
        Returns:
            List of sentence-based chunks.
        """
        max_sentence_length = 256
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If a single sentence exceeds max_length, append it separately
            if len(sentence) > max_sentence_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence.strip())
            # If adding the sentence would keep the chunk under max_length
            elif len(current_chunk) + len(sentence) + 1 <= max_sentence_length:
                current_chunk += " " + sentence if current_chunk else sentence
            # Otherwise, start a new chunk
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
def fixed_size_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    chunks = []
        
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks
    
def chunk_document(text, strategy="markdown", chunk_size=500, chunk_overlap=50):
    if strategy == "markdown":
        return markdown_header_chunks(text)
    elif strategy == "sentence":
        return sentence_chunks(text)
    elif strategy == "fixed":
        return fixed_size_chunks(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
