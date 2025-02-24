#!/usr/bin/env python
import fitz  # PyMuPDF
import re
from pathlib import Path

def is_heading(line: str, max_words: int = 12) -> bool:
    words = line.split()
    return bool(words) and len(words) <= max_words

def is_section_iii_heading(heading_text: str) -> bool:
    return heading_text.lower().strip().startswith("section iii")

def is_section_iv_heading(heading_text: str) -> bool:
    return heading_text.lower().strip().startswith("section iv")

def is_complementary_heading(heading_text: str) -> bool:
    return "complementary" in heading_text.lower()

def extract_complementary_text(pdf_path: Path, pages_to_skip: int = 5) -> str:
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
                    break
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
    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("ORACLE CONFIDENTIAL"):
            continue
        if re.match(r"^\d+\s+of\s+\d+$", line):
            continue
        
        # Remove bracket tags
        bracket_pattern = r'^\[.*?\](\s*:\s*)?'
        cleaned_line = re.sub(bracket_pattern, '', line).strip()
        
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return cleaned_lines

def merge_broken_lines(lines, merge_if_no_punct=True):
    """
    Merge lines if they don't end with standard sentence punctuation,
    EXCEPT don't merge if the next line starts with a bullet (•).
    """
    merged_lines = []
    current_paragraph = ""

    end_punct = {'.', '!', '?', ':', ';'}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # If the line starts with a bullet, treat as a new paragraph
        if line.startswith("•"):
            if current_paragraph:
                merged_lines.append(current_paragraph)
                current_paragraph = ""
            current_paragraph = line
            continue

        if not current_paragraph:
            current_paragraph = line
        else:
            last_char = current_paragraph[-1]
            if merge_if_no_punct and last_char not in end_punct:
                current_paragraph += " " + line
            else:
                merged_lines.append(current_paragraph)
                current_paragraph = line

    if current_paragraph:
        merged_lines.append(current_paragraph)

    return merged_lines

def process_pdf_to_txt(pdf_path: Path, output_path: Path, pages_to_skip: int = 5):
    extracted = extract_complementary_text(pdf_path, pages_to_skip=pages_to_skip)
    cleaned_list_of_lines = clean_extracted_text(extracted)
    merged_paragraphs = merge_broken_lines(cleaned_list_of_lines, merge_if_no_punct=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if merged_paragraphs:
            f.write("=== EXTRACTED COMPLEMENTARY CONTENT ===\n\n")
            for paragraph in merged_paragraphs:
                f.write(paragraph + "\n\n")
            f.write("=== END ===\n")
        else:
            f.write("No complementary section found (after Section III, before Section IV).\n")

def main():
    pdf_file = Path("GCP.pdf")
    out_file = Path("output.txt")
    process_pdf_to_txt(pdf_file, out_file, pages_to_skip=5)
    print(f"Done. See {out_file}")

if __name__ == "__main__":
    main()
