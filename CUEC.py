#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUEC_full.py
Extracts the "Complementary Content" section from a SOC Report PDF,
cleans up the text for neat CSV import, and exports the results as a CSV file.
Also saves the initial extracted text for debugging.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
import pandas as pd
import spacy
from collections import Counter

# Load spaCy's English model.
nlp = spacy.load("en_core_web_sm")

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

def is_junk_line(line: str) -> bool:
    """
    Determines if a line is junk using universal heuristics and spaCy NLP.
    
    Heuristics include:
      - Lines that are empty or very short (less than 3 words).
      - Lines that are entirely uppercase and short.
      - Lines containing a pipe ("|") with few words.
      - Lines that match a page marker pattern like "31 of 97".
      - Lines where a high ratio of tokens are numeric or recognized as ORG.
    """
    line = line.strip()
    if not line:
        return True
    if len(line.split()) < 3:
        return True
    if line.isupper() and len(line.split()) < 10:
        return True
    if '|' in line and len(line.split()) < 10:
        return True
    if re.fullmatch(r'\d+\s+of\s+\d+', line):
        return True
    doc = nlp(line)
    if len(doc) > 0:
        junk_token_count = sum(1 for token in doc if token.ent_type_ in {"ORG", "CARDINAL"})
        ratio = junk_token_count / len(doc)
        if ratio > 0.7 and len(doc) < 15:
            return True
    return False

def remove_emails_urls_page_numbers(line: str) -> str:
    """Remove emails, URLs, and page number patterns from the line."""
    line = re.sub(r'\S+@\S+\.\S+', '', line)       # Remove emails.
    line = re.sub(r'https?://\S+', '', line)         # Remove URLs.
    line = re.sub(r'\b\d+\s+of\s+\d+\b', '', line)    # Remove page number patterns.
    return line.strip()

def extract_complementary_text(pdf_path: Path, pages_to_skip: int = 5) -> str:
    """
    Extracts lines from the PDF (after skipping initial pages) and uses structural logic
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

def clean_extracted_text(raw_text: str) -> list:
    """
    Cleans the extracted text by:
      1. Removing emails, URLs, and page number patterns.
      2. Filtering out junk lines using universal heuristics.
      3. Stripping out any remaining bracket tags.
      4. Dynamically filtering out excessively repeated lines.
    """
    lines = raw_text.splitlines()
    temp_cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = remove_emails_urls_page_numbers(line)
        if is_junk_line(line):
            continue
        # Remove any bracket tags (e.g., [HEADING]: or [START: ...]).
        line = re.sub(r'^\[[^\]]+\]\s*:?\s*', '', line).strip()
        if line:
            temp_cleaned.append(line)
    total = len(temp_cleaned)
    threshold = max(5, int(total * 0.1))  # Remove lines appearing in >10% of total, minimum 5.
    freq = Counter(temp_cleaned)
    cleaned_lines = [line for line in temp_cleaned if freq[line] <= threshold]
    return cleaned_lines

def merge_broken_lines(lines: list) -> list:
    """
    Merges lines that are likely broken parts of the same paragraph using the following heuristic:
      - If the current line ends with a colon (:), merge it with the next line.
      - If the current line has an unmatched open parenthesis (more '(' than ')'),
        merge it with the next line.
      - Else if the current line ends with sentence-ending punctuation (., !, or ?), treat it as complete.
      - Else if the next line starts with a lowercase letter, merge them.
      - Otherwise, treat them as separate paragraphs.
    """
    merged_lines = []
    if not lines:
        return merged_lines
    current = lines[0]
    for next_line in lines[1:]:
        # If current line has an unmatched open parenthesis, merge unconditionally.
        if current.count('(') > current.count(')'):
            current = current + " " + next_line
            continue
        # If current ends with a colon, always merge.
        if current.endswith(':'):
            current = current + " " + next_line
        # Else if current ends with sentence-ending punctuation, treat as complete.
        elif current[-1] in {'.', '!', '?'}:
            merged_lines.append(current)
            current = next_line
        # Else if next line starts with a lowercase letter, merge.
        elif next_line and next_line[0].islower():
            current = current + " " + next_line
        else:
            merged_lines.append(current)
            current = next_line
    if current:
        merged_lines.append(current)
    return merged_lines

def process_pdf_to_dataframe(pdf_path: Path, pages_to_skip: int = 5) -> pd.DataFrame:
    """
    Processes the PDF by:
      - Extracting the raw text.
      - Saving the raw text to 'output.txt' for debugging.
      - Cleaning the text (using universal heuristics and dynamic frequency filtering).
      - Merging broken lines using our refined merging heuristic.
      
    Returns a DataFrame with columns 'S. No.' and 'Content'.
    """
    raw_text = extract_complementary_text(pdf_path, pages_to_skip=pages_to_skip)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    cleaned_list_of_lines = clean_extracted_text(raw_text)
    merged_paragraphs = merge_broken_lines(cleaned_list_of_lines)
    data = [{"S. No.": idx, "Content": para} for idx, para in enumerate(merged_paragraphs, start=1)]
    return pd.DataFrame(data)

def export_to_csv(df: pd.DataFrame, output_path: Path):
    """Exports the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

def main():
    pdf_file = Path("GCP.pdf")  # Change filename as needed.
    out_csv_file = Path("output.csv")
    df = process_pdf_to_dataframe(pdf_file, pages_to_skip=5)
    export_to_csv(df, out_csv_file)
    print(f"Done. See {out_csv_file} and output.txt for debugging.")

if __name__ == "__main__":
    main()
