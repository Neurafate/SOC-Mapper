# parser_app.py

import os
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import logging

# Import your parser module
import parser  # Ensure parser.py is in the same directory

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
TEXT_OUTPUT_EXTENSION = '.txt'
EXCEL_OUTPUT_EXTENSION = '.xlsx'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS
CORS(app)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/parse', methods=['POST'])
def parse_pdf():
    """
    Endpoint to parse PDF and extract text.
    Expects:
        - 'file': PDF file
        - 'start_page': Starting page number (1-based)
        - 'end_page': Ending page number (1-based)
    Returns:
        - 'txt_file': Name of the extracted text file
        - 'extracted_text': The extracted text content
    """
    logging.info("Received /parse request")
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    start_page = request.form.get('start_page', type=int)
    end_page = request.form.get('end_page', type=int)

    # Validate file
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        logging.error("Unsupported file type")
        return jsonify({'error': 'Unsupported file type. Only PDF files are allowed.'}), 400

    if not start_page or not end_page:
        logging.error("Start page and end page are required")
        return jsonify({'error': 'Start page and end page are required.'}), 400

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info(f"Saved uploaded file to {file_path}")

    # Define the output text file name
    txt_filename = f"{os.path.splitext(filename)[0]}{TEXT_OUTPUT_EXTENSION}"
    txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

    try:
        # Extract text using parser.py
        extracted_text = parser.extract_text_from_pdf(
            input_file=file_path,
            start_page=start_page,
            end_page=end_page,
            output_file=txt_file_path
        )
        logging.info(f"Extracted text and saved to {txt_file_path}")

        return jsonify({
            'txt_file': txt_filename,
            'extracted_text': extracted_text
        }), 200

    except Exception as e:
        logging.exception("Error during parsing PDF")
        return jsonify({'error': str(e)}), 500


@app.route('/chunk', methods=['POST'])
def chunk_pdf():
    """
    Endpoint to chunk PDF text based on Control IDs and generate Excel.
    Expects:
        - 'file': PDF file
        - 'start_page': Starting page number (1-based)
        - 'end_page': Ending page number (1-based)
        - 'control_id': Control IDs, comma-separated (e.g., "CC 1.2, DD 2.3")
    Returns:
        - 'excel_file': Name of the generated Excel file
    """
    logging.info("Received /chunk request")
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    start_page = request.form.get('start_page', type=int)
    end_page = request.form.get('end_page', type=int)
    control_id = request.form.get('control_id', type=str)

    # Validate file
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        logging.error("Unsupported file type")
        return jsonify({'error': 'Unsupported file type. Only PDF files are allowed.'}), 400

    if not start_page or not end_page:
        logging.error("Start page and end page are required")
        return jsonify({'error': 'Start page and end page are required.'}), 400

    if not control_id:
        logging.error("Control ID is required")
        return jsonify({'error': 'Control ID is required.'}), 400

    # Split control IDs into a list, stripping whitespace
    control_ids = [cid.strip() for cid in control_id.split(',') if cid.strip()]
    if not control_ids:
        logging.error("No valid Control IDs provided")
        return jsonify({'error': 'No valid Control IDs provided.'}), 400

    # Generate regex patterns from control IDs
    regex_patterns = [parser.generate_regex_from_sample(cid) for cid in control_ids]
    logging.info(f"Generated regex patterns: {regex_patterns}")

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info(f"Saved uploaded file to {file_path}")

    # Define intermediate and output file names
    txt_filename = f"{os.path.splitext(filename)[0]}{TEXT_OUTPUT_EXTENSION}"
    txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

    excel_filename = f"{os.path.splitext(filename)[0]}_chunked{EXCEL_OUTPUT_EXTENSION}"
    excel_file_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)

    try:
        # Step 1: Extract text
        extracted_text = parser.extract_text_from_pdf(
            input_file=file_path,
            start_page=start_page,
            end_page=end_page,
            output_file=txt_file_path
        )
        logging.info(f"Extracted text and saved to {txt_file_path}")

        # Step 2: Chunk text based on control IDs
        chunked_df = parser.chunk_text_by_multiple_patterns(
            text=extracted_text,
            patterns=regex_patterns
        )
        logging.info(f"Chunked text into {len(chunked_df)} chunks")

        # Step 3: Save chunked data to Excel
        if not chunked_df.empty:
            chunked_df.to_excel(excel_file_path, index=False)
            logging.info(f"Saved chunked data to {excel_file_path}")
        else:
            logging.warning("No chunks were created from the extracted text")

        return jsonify({
            'excel_file': excel_filename
        }), 200

    except Exception as e:
        logging.exception("Error during chunking PDF")
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<path:filename>', methods=['GET'])
def download_file(filename):
    """
    Endpoint to download files from the uploads directory.
    """
    logging.info(f"Received request to download file: {filename}")
    try:
        # Secure the filename to prevent directory traversal attacks
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

        # Check if the file exists
        if not os.path.isfile(file_path):
            logging.error(f"File not found: {file_path}")
            abort(404, description="File not found")

        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename, as_attachment=True)
    except Exception as e:
        logging.exception("Error during file download")
        abort(500, description="An error occurred while trying to download the file")


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': str(e)}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'An internal error occurred.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
