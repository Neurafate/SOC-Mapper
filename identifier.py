# identifier.py

import os
import re
import joblib
import pandas as pd
import fitz  # PyMuPDF
import string
import logging
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "logistic_regression_control_id_model.pkl"
ALLOWED_EXTENSIONS = {"pdf"}
HIGH_PRIORITY_REGEX = [
    r"AC-\d+",          # AC-01, AC-13a
    r"AC\d+\.\d+",      # AC1.1, AC2.3
    r"CC\d+\.\d+\.\d+", # CC1.1.1, CC1.1.2
    r"C\d+\.\d+\.\d+",  # C1.1.1
    r"PI\d+\.\d+",      # PI1.5
    r"OIS-\d+",         # OIS-01
    r"CC\d+\.\d+",      # CC8.1
    r"CC-\d+\.\d+",     # CC-1.1
    r"SC-\d+",          # SC-01
    r"TC\d+\.\d+",      # TC1.5
    # Add more patterns as needed
]

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Ensure NLTK is ready
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
ENGLISH_WORDS = set(words.words())  # Load a set of real English words
logger.info("Loaded NLTK words corpus and punkt tokenizer.")

def allowed_file(filename):
    """Check if the uploaded file is a valid PDF."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(input_file, start_page=1, end_page=None):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        logger.info(f"Extracting text from PDF: {input_file}")
        doc = fitz.open(input_file)
        total_pages = doc.page_count
        logger.debug(f"Total pages in PDF: {total_pages}")

        # Set default end_page if not provided
        if end_page is None:
            end_page = total_pages

        # Validate page range
        if start_page < 1 or end_page > total_pages or start_page > end_page:
            error_msg = f"Invalid page range: start_page={start_page}, end_page={end_page}, total_pages={total_pages}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        text = ""
        for page_num in range(start_page - 1, end_page):
            page = doc.load_page(page_num)
            text += page.get_text()
        logger.info(f"Extracted text from pages {start_page} to {end_page}.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise e

def generate_regex_from_sample(control_sample):
    """
    Generates a regex pattern from a control sample string.
    Handles optional parts denoted by '^'.
    This version also explicitly handles punctuation, including punctuation
    at the end of the sample (e.g., '.', ':', etc.).
    """
    logger.info(f"Generating regex from control sample: {control_sample}")
    
    # Start with a word boundary to match entire tokens
    regex = r'\b'
    
    # Split on '^' to identify optional parts
    parts = control_sample.split('^')
    
    for i, part in enumerate(parts):
        part_regex = ""
        
        # Build a regex pattern for each character in the part
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
                # Fallback: escape any other characters
                part_regex += re.escape(char)
        
        # The first part is mandatory, subsequent parts are optional
        if i == 0:
            regex += part_regex
        else:
            regex += f"(?:{part_regex})?"
    
    logger.info(f"Generated regex for sample '{control_sample}': {regex}")
    return regex

def extract_regex_patterns(control_ids):
    """
    Generates generalized regex patterns from Control IDs by replacing digits with \d+.
    """
    logger.info("Generating generalized regex patterns from Control IDs.")
    regex_patterns = {}
    
    for cid in control_ids:
        # Generate regex using generate_regex_from_sample
        pattern = generate_regex_from_sample(cid)
        regex_patterns[cid] = pattern
    
    logger.debug(f"Generated {len(regex_patterns)} generalized regex patterns.")
    return regex_patterns

def tokenize_text_into_words(text):
    """
    Tokenize text into individual words using NLTK's word tokenizer.
    """
    logger.info("Tokenizing text into individual words using NLTK's word tokenizer.")
    try:
        # Explicitly specify the language to avoid misconfiguration
        words_list = word_tokenize(text, language='english')
        logger.debug(f"Total words extracted: {len(words_list)}")
        return words_list
    except LookupError as e:
        logger.error(f"NLTK resource not found: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error during word tokenization: {e}")
        raise e

def classify_words(words_list):
    """Classify each word to identify if it's a Control ID."""
    logger.info("Classifying words to identify Control IDs.")
    results = []
    control_id_candidates = []

    for word in words_list:
        word_clean = word.strip()

        # Skip empty tokens
        if not word_clean:
            continue

        # Exclude tokens that are entirely punctuation
        if all(char in string.punctuation for char in word_clean):
            logger.debug(f"Excluded pure punctuation token: '{word_clean}'")
            continue

        # Exclude tokens with excessive repetitive punctuation (more than 3 consecutive)
        if re.search(r'([{}])\1{{3,}}'.format(re.escape(string.punctuation)), word_clean):
            logger.debug(f"Excluded repetitive punctuation token: '{word_clean}'")
            continue

        # Ensure token has a minimum number of alphanumeric characters
        alnum_count = sum(c.isalnum() for c in word_clean)
        if alnum_count < 2:
            logger.debug(f"Excluded token with insufficient alphanumerics: '{word_clean}'")
            continue

        # Filter out common English words unless structured as a Control ID
        if word_clean.lower() in ENGLISH_WORDS and not re.search(r"[\d.\-]", word_clean):
            results.append({"Word": word_clean, "Prediction": "Not a Control ID (Common Word)"})
            continue

        # Standalone Numbers Allowed (Considered Control IDs)
        if re.fullmatch(r"\d+", word_clean):
            results.append({"Word": word_clean, "Prediction": "Potential Control ID"})
            control_id_candidates.append(word_clean)
            continue

        # Reject short abbreviations without numbers
        if re.fullmatch(r"[A-Za-z]+", word_clean) and len(word_clean) < 5:
            results.append({"Word": word_clean, "Prediction": "Not a Control ID (Short Abbreviation)"})
            continue

        # Model classification
        try:
            prediction_value = model.predict([word_clean])[0]
            prediction = "Control ID" if prediction_value == 1 else "Not a Control ID"
        except Exception as e:
            logger.error(f"Model prediction error for Word '{word_clean}': {e}")
            prediction = "Prediction Error"

        # Ensure Control IDs contain both letters and numbers/special characters
        if prediction == "Control ID":
            if not re.search(r"[\d.\-]", word_clean):
                results.append({"Word": word_clean, "Prediction": "Not a Control ID (Lacks Structure)"})
                continue
            control_id_candidates.append(word_clean)

        results.append({"Word": word_clean, "Prediction": prediction})

    logger.info(f"Initial classification complete. {len(control_id_candidates)} Control ID candidates found.")

    # -------------------------------------------------------------------
    # 1) Generate regex patterns for each Control ID candidate
    # -------------------------------------------------------------------
    regex_patterns_dict = extract_regex_patterns(control_id_candidates)

    # Group Control IDs by their generalized patterns
    regex_to_words = {}
    for cid, pattern in regex_patterns_dict.items():
        if pattern not in regex_to_words:
            regex_to_words[pattern] = []
        regex_to_words[pattern].append(cid)

    # -------------------------------------------------------------------
    # 2) Apply frequency filter: keep only patterns with >= 5 occurrences
    # -------------------------------------------------------------------
    MIN_OCCURRENCES = 5
    filtered_regex_to_words = {
        pat: cids for pat, cids in regex_to_words.items() if len(cids) >= MIN_OCCURRENCES
    }

    # -------------------------------------------------------------------
    # 3) Prepare the final list of unique control IDs for preview
    # -------------------------------------------------------------------
    unique_control_ids = []
    for pattern, cids in filtered_regex_to_words.items():
        # We'll store just one "Example Control ID" from the group
        unique_control_ids.append({"Regex Pattern": pattern, "Example Control ID": cids[0]})

    # Sort regex preview → High-priority patterns first
    def priority_sort(item):
        for priority_pattern in HIGH_PRIORITY_REGEX:
            if re.fullmatch(priority_pattern, item["Example Control ID"]):
                return (0, item["Regex Pattern"])  # High priority
        return (1, item["Regex Pattern"])  # Low priority

    unique_control_ids.sort(key=priority_sort)

    logger.info(
        f"Identified {len(filtered_regex_to_words)} patterns with >= {MIN_OCCURRENCES} occurrences. "
        f"({len(regex_to_words) - len(filtered_regex_to_words)} patterns discarded)."
    )

    return unique_control_ids

def process_pdf(file_path):
    """
    Process the PDF file and return the list of repeating regex patterns.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list of dict: Each dict contains 'Regex Pattern' and 'Example Control ID'.
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if not allowed_file(file_path):
        logger.error(f"Invalid file type: {file_path}")
        raise ValueError(f"Invalid file type: {file_path}. Only PDF files are allowed.")

    try:
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)

        # Tokenize text into words
        words_list = tokenize_text_into_words(text)

        # Classify words to identify Control IDs and extract regex patterns
        regex_patterns = classify_words(words_list)

        return regex_patterns

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise e
