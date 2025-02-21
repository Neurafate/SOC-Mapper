import fitz  # PyMuPDF
from pathlib import Path

def is_heading(line: str, max_words: int = 12) -> bool:
    """
    A simple heuristic to decide if 'line' should be considered a heading:
      - The line has <= max_words.
    You can refine this (e.g., check capitalization or font size).
    """
    words = line.split()
    if not words:
        return False
    if len(words) > max_words:
        return False
    return True

def is_section_iii_heading(heading_text: str) -> bool:
    """
    True if the heading starts with 'Section III' (case-insensitive).
    Examples:
      - 'Section III'
      - 'SECTION III.'
      - 'Section III - Additional Info'
    """
    txt_lower = heading_text.lower().strip()
    return txt_lower.startswith("section iii")

def is_section_iv_heading(heading_text: str) -> bool:
    """
    True if the heading starts with 'Section IV' (case-insensitive).
    """
    txt_lower = heading_text.lower().strip()
    return txt_lower.startswith("section iv")

def is_complementary_heading(heading_text: str) -> bool:
    """
    True if 'heading_text' contains 'complementary' (case-insensitive).
    """
    return "complementary" in heading_text.lower()

def extract_complementary_text(pdf_path: Path, pages_to_skip: int = 5) -> str:
    """
    Reads the PDF with PyMuPDF.
    Skips the first `pages_to_skip` pages entirely (default = 5).
    
    Then:
      1) Looks for a heading mentioning "Section III".
      2) After that, the FIRST heading mentioning "complementary" starts capture.
      3) Capture all lines (headings or normal) until a heading mentioning "Section IV".
      4) Stop and return the captured text as a single string.
    """
    doc = fitz.open(pdf_path)
    
    all_lines = []
    
    # Skip the first 'pages_to_skip' pages
    for page_num, page in enumerate(doc, start=1):
        if page_num <= pages_to_skip:
            # ignore these pages
            continue
        
        page_text = page.get_text("text")
        for raw_line in page_text.splitlines():
            line_stripped = raw_line.strip()
            if not line_stripped:
                continue  # skip empty lines
            
            # Decide if it looks like a heading
            heading_flag = is_heading(line_stripped)
            all_lines.append({
                'text': line_stripped,
                'is_heading': heading_flag
            })
    
    # State variables
    seen_section_iii = False
    capturing = False
    captured_lines = []
    started_heading = ""
    
    for line_obj in all_lines:
        line_text = line_obj['text']
        line_is_heading = line_obj['is_heading']
        
        if line_is_heading:
            # Check if it's "Section III"
            if is_section_iii_heading(line_text):
                seen_section_iii = True
            
            # If it's "Section IV", we stop if we're currently capturing
            if is_section_iv_heading(line_text):
                if capturing:
                    # finalize
                    break
                else:
                    # not capturing yet, just ignore
                    continue
            
            # If it mentions "complementary"
            if is_complementary_heading(line_text):
                # Start capturing only if we've seen Section III
                if seen_section_iii and not capturing:
                    capturing = True
                    started_heading = line_text
                    # Include the heading in the captured text
                    captured_lines.append(f"[START: {started_heading}]")
                elif seen_section_iii and capturing:
                    # If we're already capturing, treat it as another heading within the section
                    captured_lines.append(f"[ANOTHER COMPLEMENTARY HEADING]: {line_text}")
            else:
                # Some other heading (not Section IV, not "complementary").
                if capturing:
                    captured_lines.append(f"[HEADING]: {line_text}")
        else:
            # Normal line (not a heading)
            if capturing:
                captured_lines.append(line_text)
    
    doc.close()
    
    return "\n".join(captured_lines)

def process_pdf_to_txt(pdf_path: Path, output_path: Path, pages_to_skip: int = 5):
    """
    Extract the complementary sections from a single PDF,
    skipping the first `pages_to_skip` pages,
    and write the result to a .txt file.
    """
    extracted = extract_complementary_text(pdf_path, pages_to_skip=pages_to_skip)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if extracted.strip():
            f.write("=== EXTRACTED COMPLEMENTARY CONTENT ===\n\n")
            f.write(extracted)
            f.write("\n\n=== END ===\n")
        else:
            f.write("No complementary section found (after Section III, before Section IV).\n")

if __name__ == "__main__":
    pdf_file = Path("Oracle.pdf")
    out_file = Path("output.txt")
    
    # We skip the first 5 pages. Adjust as needed.
    process_pdf_to_txt(pdf_file, out_file, pages_to_skip=5)
    print(f"Done. See {out_file}")
