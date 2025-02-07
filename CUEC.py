import os
import re
import logging
import fitz  # PyMuPDF

from flask import Flask, request, render_template_string, redirect, flash, url_for

from werkzeug.utils import secure_filename

# === Production Configuration ===
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure logging.
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed output.
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('SECRET_KEY', 'CHANGE_THIS_IN_PRODUCTION')

# === HTML Templates ===

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>Extract CUECs from SOC Report</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Extract CUECs from SOC Report</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul class="flashes">
          {% for message in messages %}
            <li>{{ message }}</li>
          {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}
    <form method="post" action="{{ url_for('process') }}" enctype="multipart/form-data">
        <label for="file">Select PDF file:</label>
        <input type="file" name="file" accept=".pdf" required>
        <br><br>
        <!-- Add a checkbox to enable debug mode if desired -->
        <label for="debug">Enable Debug Logging:</label>
        <input type="checkbox" name="debug" id="debug">
        <br><br>
        <button type="submit">Upload and Extract CUECs</button>
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html>
<head>
    <title>CUEC Extraction Results</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Extracted CUECs</h1>
    {% if cuecs %}
        {% for cuec in cuecs %}
            <h3>CUEC {{ loop.index }}</h3>
            <pre>{{ cuec }}</pre>
        {% endfor %}
    {% else %}
        <p>No CUECs were found in the uploaded document.</p>
    {% endif %}
    {% if debug_log %}
    <hr>
    <h2>Debug Log</h2>
    <pre>{{ debug_log }}</pre>
    {% endif %}
    <br>
    <a href="{{ url_for('index') }}">Back to Upload</a>
</body>
</html>
"""

# === Utility Functions ===

def allowed_file(filename):
    """Return True if the file has an allowed extension (PDF)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_full_text(pdf_path, debug_log):
    """
    Extract the complete text from a PDF document using PyMuPDF.
    Appends debug information into the provided debug_log list.
    """
    try:
        doc = fitz.open(pdf_path)
        debug_log.append(f"Opened PDF: {pdf_path} with {doc.page_count} pages.")
    except Exception as e:
        logger.error("Error opening PDF '%s': %s", pdf_path, e)
        debug_log.append(f"Error opening PDF '{pdf_path}': {e}")
        raise

    full_text = ""
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text()
        full_text += page_text + "\n"
        debug_log.append(f"Extracted text from page {i} (length {len(page_text)} chars).")
    return full_text

def extract_cuecs_from_text(full_text, debug_log):
    """
    Heuristically extract candidate CUEC segments from the full text.
    This version logs why and how it groups lines into sections.
    """
    # Expanded header patterns.
    header_pattern = re.compile(
        r'(complementary user entity controls|cuec|controls expected to be implemented|user entity controls|control objectives)',
        re.IGNORECASE
    )
    debug_log.append("Using header pattern: " + header_pattern.pattern)
    
    lines = full_text.splitlines()
    debug_log.append(f"Total lines in text: {len(lines)}")
    
    sections = []
    collecting = False
    current_section = []
    
    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        debug_log.append(f"Line {idx+1}: '{line}'")
        
        if header_pattern.search(line):
            debug_log.append(f"--> Header detected at line {idx+1}: '{line.strip()}'")
            if current_section:
                section_text = "\n".join(current_section).strip()
                debug_log.append(f"Section ended before header with {len(current_section)} lines.")
                sections.append(section_text)
                current_section = []
            collecting = True
            # Optionally log that we are skipping the header line.
            debug_log.append("Skipping header line.")
            continue

        if collecting:
            if stripped_line == "":
                # If we encounter an empty line, decide whether it terminates the section.
                if current_section and current_section[-1] == "":
                    # Two consecutive empty lines: end section.
                    debug_log.append(f"--> Ending section at line {idx+1} due to consecutive empty lines.")
                    sections.append("\n".join(current_section).strip())
                    current_section = []
                    collecting = False
                else:
                    # Append an empty line, but do not necessarily end the section.
                    debug_log.append(f"Appending empty line at line {idx+1}.")
                    current_section.append("")
                continue
            
            # Check if line looks like a bullet item or is indented.
            if re.match(r"^\s*[\-\*\•\d\.\)]", line):
                debug_log.append(f"Appending bullet/indented line at {idx+1}: '{stripped_line}'")
                current_section.append(stripped_line)
            else:
                # Otherwise, treat it as a continuation.
                if current_section:
                    debug_log.append(f"Appending continuation to previous line at {idx+1}: '{stripped_line}'")
                    current_section[-1] += " " + stripped_line
                else:
                    debug_log.append(f"Starting new section with line {idx+1}: '{stripped_line}'")
                    current_section.append(stripped_line)
    
    if current_section:
        debug_log.append("Appending final section.")
        sections.append("\n".join(current_section).strip())
    
    debug_log.append(f"Total sections detected before filtering: {len(sections)}")
    
    # Post-processing: remove sections that are too short to be genuine.
    cleaned_sections = []
    for i, sec in enumerate(sections, start=1):
        word_count = len(sec.split())
        debug_log.append(f"Section {i} word count: {word_count}")
        if word_count > 5:
            cleaned_sections.append(sec)
        else:
            debug_log.append(f"Section {i} filtered out due to insufficient length.")
    
    debug_log.append(f"Total sections after filtering: {len(cleaned_sections)}")
    return cleaned_sections

def extract_cuecs(pdf_path, debug=False):
    """
    Combine the extraction steps to retrieve candidate CUEC segments from the provided PDF.
    If debug is True, returns a tuple (cuecs, debug_log).
    """
    debug_log = []
    full_text = extract_full_text(pdf_path, debug_log)
    cuecs = extract_cuecs_from_text(full_text, debug_log)
    if debug:
        return cuecs, "\n".join(debug_log)
    else:
        return cuecs

# === Flask Routes ===

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/process', methods=['POST'])
def process():
    # Check if debug mode is enabled via checkbox (or via URL parameter)
    debug_mode = request.form.get('debug') == 'on' or request.args.get('debug') == '1'
    logger.debug("Debug mode is %s", "enabled" if debug_mode else "disabled")
    
    if 'file' not in request.files:
        flash("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == "":
        flash("No file selected for uploading.")
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash("Allowed file type is PDF.")
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        logger.debug("File saved to: %s", filepath)
    except Exception as e:
        logger.error("Error saving file '%s': %s", filename, e)
        flash("Error saving the uploaded file.")
        return redirect(request.url)
    
    try:
        if debug_mode:
            cuecs, debug_log = extract_cuecs(filepath, debug=True)
        else:
            cuecs = extract_cuecs(filepath)
            debug_log = None
    except Exception as e:
        logger.error("Error processing PDF '%s': %s", filepath, e)
        flash("Error processing the PDF.")
        return redirect(url_for('index'))
    
    return render_template_string(RESULT_HTML, cuecs=cuecs, debug_log=debug_log)

# === Entry Point ===
if __name__ == '__main__':
    # In production, run behind a WSGI server (e.g., Gunicorn).
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
