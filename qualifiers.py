# qualifiers.py

import os
import logging
import pandas as pd
from rag import retrieve_answers_for_controls, load_faiss_index
from llm_analysis import call_ollama_api
from sentence_transformers import SentenceTransformer
import datetime
import re

def qualify_soc_report(pdf_path, df_chunks_path, faiss_index_path, excel_output_path):
    """
    Performs qualifier checks on the SOC report and appends results to the Excel output.
    This function can be used for both initial checks (without saving to Excel) and final checks.
    """
    logging.info("Starting qualifier checks for SOC report.")

    try:
        # Load the RAG system and embeddings if paths are provided
        if df_chunks_path and faiss_index_path:
            model = SentenceTransformer('all-mpnet-base-v2')
            df_chunks = pd.read_csv(df_chunks_path)
            index = load_faiss_index(faiss_index_path)
        else:
            model = None
            df_chunks = None
            index = None

    except Exception as e:
        logging.error("Error loading resources: %s", e)
        return "No. The SOC 2 Type 2 Report does not provide a clear audit period duration."

    # Perform specific qualifier checks
    try:
        # Initialize results
        qualifier_results = []

        if model and index and df_chunks:
            # 1. Check if the report is latest
            latest_report_result = is_report_latest(df_chunks, model, index)
            qualifier_results.append({
                "Question": "Is the SOC 2 Type 2 Report latest (within the last 12 months)?",
                "Answer": latest_report_result
            })

            # 2. Check if trust principles are covered
            trust_principles_result = are_trust_principles_covered(df_chunks, model, index)
            qualifier_results.append({
                "Question": "Are all three Trust Principles (Security, Availability, Confidentiality) covered?",
                "Answer": trust_principles_result
            })

            # 3. Check if audit period is sufficient
            audit_period_result = is_audit_period_sufficient(df_chunks, model, index)
            qualifier_results.append({
                "Question": "Is the SOC 2 Type 2 Report covering an audit period of at least 9 months?",
                "Answer": audit_period_result
            })

            # If excel_output_path is provided, save to Excel
            if excel_output_path:
                qualifier_df = pd.DataFrame(qualifier_results)

                # Add Status column
                qualifier_df["Status"] = qualifier_df["Answer"].apply(determine_status)

                # Reorder columns: Question, Status, Answer
                qualifier_df = qualifier_df[["Question", "Status", "Answer"]]

                # Save results to the Excel file
                if os.path.exists(excel_output_path):
                    with pd.ExcelWriter(excel_output_path, mode='a', engine='openpyxl') as writer:
                        # Check if "Qualifying Questions" sheet exists
                        try:
                            existing_df = pd.read_excel(excel_output_path, sheet_name="Qualifying Questions")
                            # If exists, append without overwriting
                            startrow = existing_df.shape[0] + 1
                            qualifier_df.to_excel(
                                writer, 
                                sheet_name="Qualifying Questions", 
                                index=False, 
                                header=False, 
                                startrow=startrow
                            )
                        except ValueError:
                            # If "Qualifying Questions" sheet does not exist, create it
                            qualifier_df.to_excel(
                                writer, 
                                sheet_name="Qualifying Questions", 
                                index=False
                            )
                else:
                    with pd.ExcelWriter(excel_output_path, mode='w', engine='openpyxl') as writer:
                        qualifier_df.to_excel(
                            writer, 
                            sheet_name="Qualifying Questions", 
                            index=False
                        )
        else:
            # If model or index is not provided, perform minimal checks or skip
            logging.warning("Model, index, or df_chunks not provided. Skipping qualifier checks.")
            return "No. The SOC 2 Type 2 Report does not provide a clear audit period duration."

        # Determine overall SOC viability
        all_pass = all(qr['Status'] == "Pass" for qr in qualifier_results)
        overall_viability = "Pass" if all_pass else "Fail"
        overall_message = "Yes. The SOC is valid." if all_pass else "No. The SOC is not valid."

        return overall_message

    except Exception as e:
        logging.error("Error during qualifier checks: %s", e)
        return "No. The SOC 2 Type 2 Report does not provide a clear audit period duration."

def determine_status(answer):
    if answer.strip().startswith("Yes."):
        return "Pass"
    else:
        return "Fail"

def is_report_latest(df_chunks, model, index, top_k=3):
    """
    Determines if the SOC 2 Type 2 Report is the latest by extracting the publication date
    and comparing it with the current date.
    """
    query = "What is the publication date of the SOC 2 Type 2 Report?"
    query_emb = model.encode([query], show_progress_bar=False).astype('float32')

    # Retrieve answers from RAG
    distances, indices = index.search(query_emb, top_k)
    retrieved_answers = retrieve_answers_for_controls(
        pd.DataFrame([{'Control': query}]), model, index, df_chunks, top_k
    )

    current_date = datetime.datetime.now().date()

    prompt = f'''
    You are an expert in SOC 2 Type 2 compliance. Based solely on the following retrieved responses, determine the publication date of the SOC 2 Type 2 Report and assess whether it is the latest report.

    Retrieved Responses:
    {retrieved_answers.to_dict(orient='records')}

    Instructions:
    1. Extract the publication date of the SOC 2 Type 2 Report from the retrieved responses.
    2. Compare the extracted publication date with the current date ({current_date}).
    3. If the report was published within the last 12 months from the current date, it is considered the latest.
    4. Provide a clear and concise answer in the following exact format:
       - If the report is latest:
         "Yes. The SOC 2 Type 2 Report is the latest because it was published on [publication_date], which is within the last 12 months."
       - If the report is not latest:
         "No. The SOC 2 Type 2 Report is not the latest because it was published on [publication_date], which is more than 12 months ago."
    5. Do not include any additional information or commentary beyond the specified format.
    '''

    logging.debug("Prompt for 'is_report_latest': %s", prompt)

    response = call_ollama_api(prompt)
    return response

def are_trust_principles_covered(df_chunks, model, index, top_k=3):
    """
    Checks if the SOC 2 Type 2 Report covers the three Trust Principles: Security, Availability, and Confidentiality.
    """
    query = "Does the SOC 2 Type 2 Report cover the principles of Security, Availability, and Confidentiality?"
    query_emb = model.encode([query], show_progress_bar=False).astype('float32')

    # Retrieve answers from RAG
    distances, indices = index.search(query_emb, top_k)
    retrieved_answers = retrieve_answers_for_controls(
        pd.DataFrame([{'Control': query}]), model, index, df_chunks, top_k
    )

    prompt = f'''
    You are an expert in SOC 2 Type 2 compliance. Based solely on the following retrieved responses, determine whether the SOC 2 Type 2 Report covers all three Trust Principles: Security, Availability, and Confidentiality.

    Retrieved Responses:
    {retrieved_answers.to_dict(orient='records')}

    Instructions:
    1. Analyze the retrieved responses to identify mentions of the following Trust Principles:
       - Security
       - Availability
       - Confidentiality
    2. Determine if all three principles are explicitly covered in the SOC 2 Type 2 Report.
    3. Provide a clear and concise answer in the following exact format:
       - If all principles are covered:
         "Yes. The SOC 2 Type 2 Report covers all three Trust Principles (Security, Availability, Confidentiality) because [specific reasons]."
       - If any principle is not covered:
         "No. The SOC 2 Type 2 Report does not cover all three Trust Principles (Security, Availability, Confidentiality) because [specific reasons]."
    4. Ensure that the reasoning specifically addresses each principle mentioned or omitted.
    5. Do not include any additional information or commentary beyond the specified format.
    '''

    logging.debug("Prompt for 'are_trust_principles_covered': %s", prompt)

    response = call_ollama_api(prompt)
    return response

def is_audit_period_sufficient(df_chunks, model, index, top_k=3):
    """
    Checks if the SOC 2 Type 2 Report covers an audit period of at least 9 months.
    """
    query = "Does the SOC 2 Type 2 Report cover an audit period of at least 9 months?"
    query_emb = model.encode([query], show_progress_bar=False).astype('float32')

    # Retrieve answers from RAG
    distances, indices = index.search(query_emb, top_k)
    retrieved_answers = retrieve_answers_for_controls(
        pd.DataFrame([{'Control': query}]), model, index, df_chunks, top_k
    )

    prompt = f'''
    You are an expert in SOC 2 Type 2 compliance. Based solely on the following retrieved responses, determine whether the audit period covered in the SOC 2 Type 2 Report is at least 9 months.

    Retrieved Responses:
    {retrieved_answers.to_dict(orient='records')}

    Instructions:
    1. Extract the start and end dates of the audit period from the retrieved responses.
    2. Calculate the total duration of the audit period in months.
       - Consider partial months as full months for simplicity (e.g., from February 15 to May 14 is considered 3 months).
    3. If the audit period is 9 months or longer, it is considered sufficient.
    4. Provide a clear and concise answer in the following exact format:
       - If the audit period is sufficient:
         "Yes. The SOC 2 Type 2 Report covers an audit period of [duration] months, which meets the requirement of at least 9 months."
       - If the audit period is insufficient:
         "No. The SOC 2 Type 2 Report covers an audit period of [duration] months, which does not meet the requirement of at least 9 months."
    5. Do not include any additional information or commentary beyond the specified format.

    Example:
    Retrieved Responses:
    ["Response": "For the Period February 1, 2023 to May 31, 2023"]

    Analysis:
    - Start date: February 1, 2023
    - End date: May 31, 2023
    - Duration: 4 months

    Answer:
    "No. The SOC 2 Type 2 Report covers an audit period of 4 months, which does not meet the requirement of at least 9 months."
    '''

    logging.debug("Prompt for 'is_audit_period_sufficient': %s", prompt)

    response = call_ollama_api(prompt)

    # Automated Parsing of the Response
    match = re.search(r'covers an audit period of (\d+) months', response)
    if match:
        duration = int(match.group(1))
        if duration >= 9:
            expected_response = f"Yes. The SOC 2 Type 2 Report covers an audit period of {duration} months, which meets the requirement of at least 9 months."
        else:
            expected_response = f"No. The SOC 2 Type 2 Report covers an audit period of {duration} months, which does not meet the requirement of at least 9 months."

        # Validate the LLM's response matches the expected response
        if response.strip() == expected_response:
            return response
        else:
            logging.warning("LLM response does not match expected format. Response: %s", response)
            return expected_response
    else:
        logging.error("Failed to parse the LLM response for audit period duration.")
        return "No. The SOC 2 Type 2 Report does not provide a clear audit period duration."

# -------------------------------------------------------
# NEW ENDPOINT: /qualify_soc_report
# -------------------------------------------------------
@app.route('/qualify_soc_report', methods=['POST'])
def qualify_soc_report_endpoint():
    """
    Endpoint to perform qualifier checks at the start of processing.
    This allows the user to decide whether to proceed based on the qualifiers.
    """
    logging.info("Received request to /qualify_soc_report endpoint.")
    try:
        pdf_file = request.files.get('pdf_file')
        if not pdf_file:
            raise ValueError("Missing required file: pdf_file")

        # Secure filename and save
        filename_pdf = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_pdf)
        pdf_file.save(pdf_path)
        logging.info(f"PDF file saved to {pdf_path}")

        # Generate a unique task ID for qualifier checks
        task_id = str(uuid.uuid4())
        progress_data[task_id] = {
            'progress': 0.0,
            'status': 'Performing initial qualifier checks...',
            'download_url': None,
            'error': None,
            'eta': 30,  # Estimated time for qualifiers
            'cancelled': False
        }

        # Start background thread for qualifier checks
        thread = threading.Thread(
            target=background_qualifier_process,
            args=(task_id, pdf_path)
        )
        thread.start()

        return jsonify({"message": "Qualifier checks started", "task_id": task_id}), 202

    except Exception as e:
        logging.error(f"Error in /qualify_soc_report endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400

def background_qualifier_process(task_id, pdf_path):
    """
    Background process to perform qualifier checks at the start.
    """
    try:
        logging.info(f"Background qualifier processing started for task_id: {task_id}.")

        # Perform qualifier checks without saving to Excel
        with progress_lock:
            progress_data[task_id]['status'] = "Performing initial qualifier checks..."
        logging.info(f"Task {task_id}: Performing initial qualifier checks.")

        # Perform qualifier checks
        qualifier_result = qualify_soc_report(
            pdf_path=pdf_path,
            df_chunks_path=None,          # No need to save chunks for initial check
            faiss_index_path=None,        # No need to build index for initial check
            excel_output_path=None        # Do not save to Excel
        )

        # Determine Pass/Fail based on qualifier_result
        if "Yes." in qualifier_result:
            status = "Pass"
            message = "The SOC 2 Type 2 Report is valid."
        else:
            status = "Fail"
            message = "The SOC 2 Type 2 Report is not valid."

        with progress_lock:
            progress_data[task_id]['status'] = message
            progress_data[task_id]['progress'] = 100.0
            progress_data[task_id]['download_url'] = None

        logging.info(f"Task {task_id}: Qualifier checks completed with status: {status}")

    except Exception as e:
        logging.error(f"Error in background_qualifier_process (task_id: {task_id}): {e}", exc_info=True)
        with progress_lock:
            progress_data[task_id]['status'] = "Error occurred during qualifier checks."
            progress_data[task_id]['error'] = str(e)
            progress_data[task_id]['progress'] = 0.0
            progress_data[task_id]['eta'] = 0

# -------------------------------------------------------
# Existing Endpoint: /process_all
# -------------------------------------------------------
@app.route('/process_all', methods=['POST'])
def process_all_endpoint():
    logging.info("Received request to /process_all endpoint.")
    try:
        pdf_file = request.files.get('pdf_file')
        excel_file = request.files.get('excel_file')

        if not pdf_file or not excel_file:
            raise ValueError("Missing required files: pdf_file or excel_file")

        # Control IDs are mandatory
        control_id = request.form.get('control_id', '').strip()
        if not control_id:
            logging.error("No Control IDs provided in the request.")
            return jsonify({"error": "No Control IDs were provided."}), 400

        model_name = request.form.get('model_name', 'llama3.1')
        logging.info(f"Selected model: {model_name}")

        # Secure filenames
        filename_pdf = secure_filename(pdf_file.filename)
        filename_excel = secure_filename(excel_file.filename)

        # Save uploaded files
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_pdf)
        pdf_file.save(pdf_path)
        logging.info(f"PDF file saved to {pdf_path}")

        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_excel)
        excel_file.save(excel_path)
        logging.info(f"Excel file saved to {excel_path}")

        # Detect Control IDs and their pages
        # To handle shared regex patterns, group Control IDs by their regex
        repeating_patterns = identify_control_ids(pdf_path)  # List of dicts with 'Regex Pattern' and 'Example Control ID'
        regex_to_cids = {}
        for pattern_dict in repeating_patterns:
            regex = pattern_dict.get("Regex Pattern")
            cid = pattern_dict.get("Example Control ID")
            if regex and cid and cid in [c.strip() for c in control_id.split(',') if c.strip()]:
                regex_to_cids.setdefault(regex, []).append(cid)

        # Extract Control IDs from repeating patterns (filtered)
        control_ids_order = [cid.strip() for cid in control_id.split(',') if cid.strip()]
        logging.info(f"Control IDs order for page range determination: {control_ids_order}")

        if not regex_to_cids:
            raise ValueError("No matching Control IDs found in the document based on the selected Control IDs.")

        control_id_pages = detect_control_id_pages(pdf_path, regex_to_cids)

        # Determine page range based on Control IDs with prioritization
        start_page, end_page = determine_page_range(control_id_pages, regex_to_cids)
        logging.info(f"Determined page range based on Control IDs: Start Page = {start_page}, End Page = {end_page}")

        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        progress_data[task_id] = {
            'progress': 0.0,
            'status': 'Starting task...',
            'download_url': None,
            'error': None,
            'start_time': time.time(),
            'eta': None,
            'num_controls': 0,
            'cancelled': False
        }

        # Extract base filenames for final output naming
        soc_report_filename = filename_pdf
        framework_filename = filename_excel

        # Start background processing thread with determined start and end pages
        thread = threading.Thread(
            target=background_process,
            args=(task_id, pdf_path, excel_path, start_page, end_page, control_id, model_name, soc_report_filename, framework_filename)
        )
        thread.start()

        return jsonify({"message": "Processing started", "task_id": task_id}), 202

    except Exception as e:
        logging.error(f"Error in /process_all endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# -------------------------------------------------------
# Existing Endpoint: /progress/<task_id>
# -------------------------------------------------------
@app.route('/progress/<task_id>', methods=['GET'])
def sse_progress(task_id):
    def generate():
        while True:
            task_info = progress_data.get(task_id, None)
            if task_info is None:
                yield f"data: {json.dumps({'error': 'Invalid task ID'})}\n\n"
                break

            progress = task_info['progress']
            status = task_info['status']
            eta = task_info.get('eta', None)
            error = task_info.get('error', None)
            download_url = task_info.get('download_url', None)
            cancelled = task_info.get('cancelled', False)

            data = {
                'progress': round(progress, 2),
                'status': status,
                'eta': round(eta, 2) if eta is not None else eta
            }
            if download_url:
                data['download_url'] = download_url
            if error:
                data['error'] = error
            if cancelled:
                data['cancelled'] = True

            yield f"data: {json.dumps(data)}\n\n"

            # Stop streaming if finished, cancelled, or error encountered
            if progress >= 100 or error or cancelled:
                break

            time.sleep(1)

    response = Response(generate(), mimetype='text/event-stream')

    # ✅ Add required headers to prevent buffering issues
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'  # ✅ Prevents disconnects
    response.headers['X-Accel-Buffering'] = 'no'  # ✅ Disables buffering

    return response


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    Endpoint to download a file from the specified directory.
    Restricts access to 'excel_outputs' directory for security.
    """
    logging.info(f"Received request to download file: {filename}")
    try:
        # Ensure the filename is secure
        filename = secure_filename(filename)
        directory = app.config['EXCEL_FOLDER']
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            logging.error(f"File not found: {filename}")
            return jsonify({"error": "File not found"}), 404
        return send_file(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True
        )

    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}", exc_info=True)
        return jsonify({"error": "File could not be served"}), 500

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """
    Endpoint to signal that the user wants to cancel a running task.
    The background process will check the `cancelled` flag and halt.
    """
    task_info = progress_data.get(task_id)
    if not task_info:
        logging.warning(f"Attempted to cancel invalid task_id: {task_id}.")
        return jsonify({"error": "Invalid task ID"}), 404

    if task_info.get('cancelled'):
        logging.info(f"Task {task_id} is already cancelled.")
        return jsonify({"message": f"Task {task_id} is already cancelled."}), 200

    # Mark the task as cancelled
    task_info['cancelled'] = True
    task_info['status'] = "Cancelled"
    task_info['progress'] = 0
    task_info['eta'] = 0
    task_info['error'] = "Task was cancelled by the user."

    logging.info(f"Task {task_id} marked as cancelled by the user.")
    return jsonify({"message": f"Task {task_id} cancelled."}), 200

if __name__ == "__main__":
    logging.info("Starting the Flask application on port 5000.")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
