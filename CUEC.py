#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUEC_full.py
Extracts the "Complementary Content" section from a SOC Report PDF,
applies a refined filtering pipeline using spaCy, an LLM (phi4 via Ollama), 
and Pydantic to clean and validate the text, and then exports the results as a CSV file.
Also saves the initial extracted text for debugging.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
import pandas as pd
import spacy
from collections import Counter
import ollama
from pydantic import BaseModel, validator

# ---------------------------
# Existing PDF Extraction Functions
# ---------------------------
# (These functions remain unchanged so that the extraction logic based on headings is preserved.)
nlp_spacy = spacy.load("en_core_web_sm")

def is_heading(line: str, max_words: int = 12) -> bool:
    """Determine if a line qualifies as a heading based on word count."""
    words = line.split()
    return bool(words) and len(words) <= max_words

def is_section_iii_heading(heading_text: str) -> bool:
    """Check if a heading corresponds to Section III."""
    return heading_text.lower().strip().startswith("section iii")

def is_section_iv_heading(heading_text: str) -> bool:
    """Check if a heading corresponds to Section IV."""
    return heading_text.lower().strip().startswith("section iv")

def is_complementary_heading(heading_text: str) -> bool:
    """Check if a heading is the complementary content heading."""
    return "complementary" in heading_text.lower()

def extract_complementary_text(pdf_path: Path, pages_to_skip: int = 5) -> str:
    """
    Extracts lines from the PDF (after skipping initial pages) using structural logic
    (via heading markers) to capture the complementary content section.
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    for page_num, page in enumerate(doc, start=1):
        if page_num <= pages_to_skip:
            continue
        page_text = page.get_text("text")
        for raw_line in page_text.splitlines():
            line_stripped = raw_line.strip()
            if not line_stripped:
                continue
            heading_flag = is_heading(line_stripped)
            all_lines.append({'text': line_stripped, 'is_heading': heading_flag})
    doc.close()
    
    seen_section_iii = False
    capturing = False
    captured_lines = []
    for line_obj in all_lines:
        line_text = line_obj['text']
        line_is_heading = line_obj['is_heading']
        if line_is_heading:
            if is_section_iii_heading(line_text):
                seen_section_iii = True
            if is_section_iv_heading(line_text):
                if capturing:
                    break  # End capturing once Section IV is reached.
                else:
                    continue
            if is_complementary_heading(line_text):
                if seen_section_iii and not capturing:
                    capturing = True
                    captured_lines.append(f"[START: {line_text}]")
                elif seen_section_iii and capturing:
                    captured_lines.append(f"[ANOTHER COMPLEMENTARY HEADING]: {line_text}")
            else:
                if capturing:
                    captured_lines.append(f"[HEADING]: {line_text}")
        else:
            if capturing:
                captured_lines.append(line_text)
    return "\n".join(captured_lines)

# ---------------------------
# New Filtering Pipeline Functions
# ---------------------------
def spacy_prefilter(text: str) -> str:
    """
    Pre-filter the text using spaCy:
      - Remove tags enclosed in square brackets (e.g., [HEADING], [START], etc.).
      - Split text into sentences.
      - Remove duplicate sentences and sentences that are isolated tag words.
      - Join the sentences into a single cleaned paragraph.
    """
    # Remove any content within square brackets.
    text = re.sub(r"\[.*?\]", "", text)
    
    # Process text with spaCy.
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Extract and clean sentences.
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Remove sentences that are just common tag words.
    tag_words = {"start", "heading", "complementary", "another"}
    filtered_sentences = [sent for sent in sentences if sent.lower() not in tag_words]
    
    # Remove duplicate sentences (case-insensitive).
    unique_sentences = []
    seen = set()
    for sent in filtered_sentences:
        low = sent.lower()
        if low not in seen:
            seen.add(low)
            unique_sentences.append(sent)
    
    cleaned_text = " ".join(unique_sentences)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def get_cleaned_output(text: str) -> str:
    """
    Sends the pre-filtered text to the phi4 model via Ollama with a prompt instructing it
    to return only the cleaned, plain text (without extra commentary).
    """
    filtered_text = spacy_prefilter(text)
    prompt = (
        "Please clean up the following text. Do not add any summaries, commentary, or extra explanations—"
        "return only the cleaned, plain text preserving its original structure:\n\n"
        + filtered_text
    )
    result = ollama.generate(model="phi4", prompt=prompt)
    return result["response"]

# ---------------------------
# Pydantic Validation
# ---------------------------
class CleanedLine(BaseModel):
    line: str

    @validator("line")
    def non_empty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Line cannot be empty")
        return value

def filter_and_validate_lines(text: str) -> list:
    """
    Splits the cleaned text into individual lines and validates each line using Pydantic.
    Returns a list of dictionaries with validated lines.
    """
    raw_lines = text.splitlines()
    valid_entries = []
    for line in raw_lines:
        if line.strip():
            try:
                entry = CleanedLine(line=line)
                valid_entries.append(entry.dict())
            except Exception as e:
                print(f"Skipping line: {line}\nError: {e}")
    return valid_entries

# ---------------------------
# New PDF Processing Function
# ---------------------------
def process_pdf_to_dataframe(pdf_path: Path, pages_to_skip: int = 5) -> pd.DataFrame:
    """
    Processes the PDF by:
      - Extracting the raw complementary text (using the original extraction logic).
      - Saving the raw text to 'output.txt' for debugging.
      - Cleaning the extracted text using the new filtering pipeline (spaCy prefilter + phi4 model via Ollama).
      - Validating the cleaned text line-by-line using Pydantic.
      - Returning a DataFrame with columns 'S. No.' and 'Content'.
    """
    # Extract raw complementary text from the PDF.
    raw_text = extract_complementary_text(pdf_path, pages_to_skip=pages_to_skip)
    
    # Save raw extracted text for debugging.
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    # Apply the new filtering pipeline to get the cleaned text.
    cleaned_output = get_cleaned_output(raw_text)
    
    # Validate and filter cleaned output lines.
    validated_lines = filter_and_validate_lines(cleaned_output)
    
    # Create a DataFrame (each validated line becomes a row).
    data = [{"S. No.": idx, "Content": entry["line"]} for idx, entry in enumerate(validated_lines, start=1)]
    df = pd.DataFrame(data)
    return df

def export_to_csv(df: pd.DataFrame, output_path: Path):
    """Exports the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

# ---------------------------
# Main Function for Production Use
# ---------------------------
def main():
    pdf_file = Path("GCP.pdf")  # Adjust the filename/path as needed.
    out_csv_file = Path("output.csv")
    df = process_pdf_to_dataframe(pdf_file, pages_to_skip=5)
    export_to_csv(df, out_csv_file)
    print(f"Done. See '{out_csv_file}' and 'output.txt' for debugging.")

if __name__ == "__main__":
    main()
