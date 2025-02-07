#!/usr/bin/env python
"""
rag_module.py

This module processes a PDF file in reverse page order, sending each page’s text to the Ollama API
to check for evidence of Complementary Controls (CUECs). The API call is now structured similarly to
the working example in llm_analysis.py.
"""

import fitz  # PyMuPDF
import requests
import logging
import sys
import json
import uuid

# Configuration for the Ollama API
OLLAMA_API_URL = "http://localhost:11434"   # Base URL; endpoint appended in the function
LLAMA_MODEL_NAME = "llama3.1"                 # Update if different
MAX_TOKENS = 1024                           # Increased from 512 to 1024

def call_ollama_api(chunk_text, model=LLAMA_MODEL_NAME, max_tokens=MAX_TOKENS):
    """
    Sends a text chunk to the Ollama API to determine whether it contains Complementary Controls (CUECs).
    Uses a structured payload similar to llm_analysis.py and streams the response to collect generated text.
    
    Parameters:
        chunk_text (str): The text to analyze.
        model (str): The model name to use.
        max_tokens (int): Maximum tokens allowed in the response.
        
    Returns:
        str or None: The generated response text from the API or None on error.
    """
    logging.info("Calling Ollama API for CUEC analysis.")
    
    # Build the prompt by appending the chunk text
    prompt = (
        "You are an expert reviewer for SOC2 Type2 reports. "
        "Examine the text below and determine if it contains evidence of Complementary Controls (CUECs). "
        "Respond with a concise 'Yes' or 'No' along with a brief explanation if CUECs are found.\n\n"
        "Text:\n" + chunk_text
    )
    
    # Construct the full API URL
    url = f"{OLLAMA_API_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    session_id = str(uuid.uuid4())
    
    payload = {
        "model": model,
        "prompt": prompt,
        "session_id": session_id,
        "num_ctx": 2048,
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=120)
        response.raise_for_status()
        generated_text = ''
        # Process the streaming response line by line
        for line in response.iter_lines():
            if line:
                try:
                    line_json = json.loads(line.decode('utf-8').strip())
                    token = line_json.get("response", "")
                    generated_text += token
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON line from API.")
        logging.debug("Ollama API generated text: %s", generated_text.strip())
        return generated_text.strip()
    except Exception as e:
        logging.error("Error calling Ollama API: %s", e)
        return None

def process_pdf_pages_reverse(document_path, start_page):
    """
    Processes the PDF in reverse order starting from `start_page` down to page 1.
    For each page, the text is extracted and sent to the Ollama API.
    
    Parameters:
        document_path (str): The path to the PDF document.
        start_page (int): The starting page number for the reverse search.
        
    Returns:
        List[dict]: A list of dictionaries containing page number, extracted text, and API response.
    """
    try:
        doc = fitz.open(document_path)
    except Exception as e:
        logging.error("Failed to open PDF file: %s", e)
        sys.exit(1)
    
    total_pages = doc.page_count
    if start_page < 1 or start_page > total_pages:
        logging.error("Invalid starting page: %d. The document has %d pages.", start_page, total_pages)
        sys.exit(1)
    
    results = []
    # Loop in reverse order (0-based indexing in PyMuPDF)
    for page_num in range(start_page - 1, -1, -1):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        logging.info("Processing page %d of %d (reverse order)", page_num + 1, total_pages)
        
        # Send the entire page text to the API
        cuec_response = call_ollama_api(page_text)
        
        results.append({
            "page": page_num + 1,
            "text": page_text,
            "cuec_response": cuec_response,
        })
    
    return results

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Expect two command-line arguments: document path and starting page
    if len(sys.argv) != 3:
        logging.error("Usage: python rag_module.py <document_path> <starting_page>")
        sys.exit(1)
    
    document_path = sys.argv[1]
    try:
        starting_page = int(sys.argv[2])
    except ValueError:
        logging.error("Starting page must be an integer.")
        sys.exit(1)
    
    results = process_pdf_pages_reverse(document_path, starting_page)
    
    # Display the results for each page
    for result in results:
        page = result["page"]
        cuec_response = result["cuec_response"]
        if cuec_response:
            logging.info("Page %d: API Response: %s", page, cuec_response)
        else:
            logging.warning("Page %d: No response from Ollama API.", page)

if __name__ == "__main__":
    main()
