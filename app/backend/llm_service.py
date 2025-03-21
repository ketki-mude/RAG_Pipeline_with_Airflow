import os
from typing import Dict, Any, Tuple, Optional
import litellm
import tiktoken
from google.generativeai import configure, GenerativeModel

def generate_response_with_gemini(query, context=None, model_name="gemini-1.5-pro"):
    """Generate a response using Google Gemini model"""
    try:
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: Google API key not configured. Please set the GOOGLE_API_KEY environment variable.", None
        
        configure(api_key=api_key)
        model = GenerativeModel(model_name)
        
        # For token counting
        encoder = tiktoken.get_encoding("cl100k_base")
        
        if context:
            prompt = f"""
            Please answer the following question based only on the provided context. 
            If the context doesn't contain the information needed to answer the question, say that you don't have enough information.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
        else:
            prompt = query
        
        # Count input tokens before API call
        input_tokens = len(encoder.encode(prompt))
        
        # Generate response
        response = model.generate_content(prompt)
        answer_text = response.text
        
        # Count output tokens
        output_tokens = len(encoder.encode(answer_text))
        
        # Calculate costs based on pricing for Gemini models
        # Using approximate pricing: $0.0000125 per 1K input tokens, $0.00005 per 1K output tokens
        input_cost = (input_tokens / 1000) * 0.0000125
        output_cost = (output_tokens / 1000) * 0.00005
        total_cost = input_cost + output_cost
        
        # Return token and cost info alongside the answer
        token_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model_name
        }
        
        return answer_text, token_info
    except Exception as e:
        return f"Error generating response: {str(e)}", None