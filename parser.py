# parser.py

import os
import re
import logging
import pandas as pd
import fitz  # PyMuPDF
import string

def extract_text_from_pdf(input_file, start_page, end_page, output_file):
    logging.info(f"Extracting text from PDF: {input_file}, pages {start_page}-{end_page}")
    
    doc = fitz.open(input_file)
    text = ""
    total_pages = doc.page_count
    # Validate page range
    if start_page < 1 or end_page > total_pages or start_page > end_page:
        error_msg = f"Invalid page range: start_page={start_page}, end_page={end_page}, total_pages={total_pages}"
        logging.error(error_msg)
        return f"An error occurred: {error_msg}"
    
    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        text += page.get_text()
    with open(output_file, "w", encoding="utf-8") as output:
        output.write(text)
    logging.info(f"Text extracted and saved to {output_file}")
    return text

def extract_tables_with_pymupdf(input_file, start_page, end_page):
    logging.info(f"Extracting tables with PyMuPDF from PDF: {input_file}, pages {start_page}-{end_page}")
    tables = []
    
    doc = fitz.open(input_file)
    total_pages = doc.page_count
    # Validate page range
    if start_page < 1 or end_page > total_pages or start_page > end_page:
        error_msg = f"Invalid page range: start_page={start_page}, end_page={end_page}, total_pages={total_pages}"
        logging.error(error_msg)
        return []
    
    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # Text block
                table = []
                current_row = []
                last_y = None
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text'].strip()
                        y = span['bbox'][1]
                        if last_y is None or abs(y - last_y) < 10:  # Adjust tolerance as needed
                            current_row.append(text)
                        else:
                            if current_row:
                                table.append(current_row)
                            current_row = [text]
                        last_y = y
                if current_row:
                    table.append(current_row)
                if table:
                    # Convert to DataFrame
                    if len(table) >= 2:
                        df = pd.DataFrame(table[1:], columns=table[0])  # Assuming first row is header
                        tables.append(df)
                        logging.debug(f"Extracted table with columns: {table[0]}")
    logging.info(f"Extracted {len(tables)} tables.")
    return tables

def chunk_text_by_multiple_patterns(text, patterns):
    """
    Chunk text based on multiple regex patterns.
    Each chunk runs from the current matched pattern
    until the next matched pattern (or end of text).
    Then we optionally trim the chunk at the last period.
    """
    logging.info("Chunking text by patterns (with optional final period trim).")
    chunks = []
    
    # Build a combined pattern that anchors on word boundaries (\b).
    # If you have more complex patterns, you can adapt it to your needs—
    # e.g., '^' logic, or custom function to build these patterns.
    combined_pattern = r'\b(?:' + '|'.join(patterns) + r')'

    # 1) Iterate over all pattern matches
    all_matches = list(re.finditer(combined_pattern, text, re.MULTILINE))
    
    for i, match in enumerate(all_matches):
        control_id = match.group()
        
        # Start of this chunk is the START of the matched pattern
        # so we INCLUDE the matched string (e.g. "AC-05:")
        chunk_start = match.start()
        
        # By default, go to the END of the text
        chunk_end = len(text)
        
        # 2) Look for next pattern after the current match
        if i < len(all_matches) - 1:
            next_match_start = all_matches[i+1].start()
            chunk_end = next_match_start
        
        # Extract the chunk
        chunk = text[chunk_start:chunk_end].strip()
        
        # 3) (Optional) Trim the chunk at the final period in multiline mode.
        #    This is what your old snippet did:
        #    eol_match = re.search(r'\.\s*$', chunk, re.MULTILINE)
        #    If we want to trim after the *last* period in that chunk, 
        #    we can do a find-all and pick the last one or a different strategy.
        #    For a direct port from your old code:
        eol_match = re.search(r'\.\s*$', chunk, re.MULTILINE)
        if eol_match:
            chunk = chunk[: eol_match.end()]

        # 4) Save the chunk
        chunks.append({
            "Control ID": control_id,
            "Content": chunk
        })
    
    logging.info(f"Created {len(chunks)} text chunks.")
    return pd.DataFrame(chunks)

def generate_regex_from_sample(control_sample):
    """
    Generates a regex pattern from a control sample string.
    Handles optional parts denoted by '^'.
    This version also explicitly handles punctuation, including punctuation
    at the end of the sample (e.g. '.', ':', etc.).
    """
    logging.info(f"Generating regex from control sample: {control_sample}")
    
    # Start with a word boundary so that we match entire tokens
    regex = r'\b'
    
    # Split on '^' to identify optional parts
    parts = control_sample.split('^')
    
    for i, part in enumerate(parts):
        part_regex = ""
        
        # Build a small per-character pattern
        for char in part:
            if char.isupper():
                part_regex += "[A-Z]"
            elif char.islower():
                part_regex += "[a-z]"
            elif char.isdigit():
                part_regex += r"\d"
            # Handle punctuation and whitespace
            elif char in string.punctuation or char.isspace():
                part_regex += re.escape(char)
            else:
                # Fallback: escape anything else
                part_regex += re.escape(char)
        
        # The first part is mandatory, subsequent parts (if any) are optional
        if i == 0:
            regex += part_regex
        else:
            regex += f"(?:{part_regex})?"
    
    # Remove the trailing word boundary to include punctuation
    # If you prefer to keep the word boundary at the end when no punctuation is present,
    # you can make the punctuation optional.
    # For example: regex += r'(?:\b|(?=\s))'
    # Here, we'll remove it.
    # regex += r'\b'  # Removed
    
    logging.info(f"Generated regex for sample '{control_sample}': {regex}")
    return regex

def concatenate_domain_control(df):
    """
    Concatenates 'Sub-Domain' and 'Control' columns to form 'Domain_Control'.
    """
    logging.info("Concatenating 'Sub-Domain' and 'Control' to form 'Domain_Control'.")
    df['Domain_Control'] = df.apply(lambda row: f"{row.get('Sub-Domain', '')}: {row.get('Control', '')}", axis=1)
    logging.info("'Domain_Control' column created successfully.")
    return df

def chunk_tables(tables):
    """
    Preprocess tables to ensure consistency before combining with text.
    """
    logging.info("Preprocessing extracted tables.")
    processed_tables = []
    for df in tables:
        # Example preprocessing: Remove empty rows/columns
        df_clean = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        # Reset index after dropping
        df_clean.reset_index(drop=True, inplace=True)
        # Additional preprocessing steps can be added here
        processed_tables.append(df_clean)
        logging.debug(f"Processed table with shape: {df_clean.shape}")
    return processed_tables
