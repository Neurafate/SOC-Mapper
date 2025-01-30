# qualifiers.py

import os
import logging
import pandas as pd
import datetime
import re
from rag import retrieve_answers_for_controls, load_faiss_index
from llm_analysis import call_ollama_api
from sentence_transformers import SentenceTransformer

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

def qualify_soc_report(pdf_path, df_chunks_path, faiss_index_path, excel_output_path):
    """
    Performs qualifier checks on the SOC report and appends results to the Excel output.
    """
    logging.info("Starting qualifier checks for SOC report.")

    try:
        # Load the RAG system and embeddings
        model = SentenceTransformer('all-mpnet-base-v2')
        df_chunks = pd.read_csv(df_chunks_path)
        index = load_faiss_index(faiss_index_path)
    except Exception as e:
        logging.error("Error loading resources: %s", e)
        return

    # Perform specific qualifier checks
    try:
        latest_report_result = is_report_latest(df_chunks, model, index)
        trust_principles_result = are_trust_principles_covered(df_chunks, model, index)
        audit_period_result = is_audit_period_sufficient(df_chunks, model, index)
    except Exception as e:
        logging.error("Error during qualifier checks: %s", e)
        return

    # Compile results
    qualifier_results = [
        {
            "Question": "Is the SOC 2 Type 2 Report latest (within the last 12 months)?",
            "Answer": latest_report_result
        },
        {
            "Question": "Are all three Trust Principles (Security, Availability, Confidentiality) covered?",
            "Answer": trust_principles_result
        },
        {
            "Question": "Is the SOC 2 Type 2 Report covering an audit period of at least 9 months?",
            "Answer": audit_period_result
        }
    ]

    # Convert to DataFrame
    qualifier_df = pd.DataFrame(qualifier_results)

    # Define function to determine Pass/Fail
    def determine_status(answer):
        if answer.strip().startswith("Yes."):
            return "Pass"
        else:
            return "Fail"

    # Add Status column
    qualifier_df["Status"] = qualifier_df["Answer"].apply(determine_status)

    # Reorder columns: Question, Status, Answer
    qualifier_df = qualifier_df[["Question", "Status", "Answer"]]

    # Save results to the Excel file
    try:
        if os.path.exists(excel_output_path):
            # Load existing workbook
            wb = load_workbook(excel_output_path)
            if "Qualifying Questions" in wb.sheetnames:
                ws = wb["Qualifying Questions"]
                # Find the last row
                last_row = ws.max_row
                # Append new data without headers
                for index, row in qualifier_df.iterrows():
                    ws.append(row.tolist())
                # Save workbook after appending
                wb.save(excel_output_path)
            else:
                # If "Qualifying Questions" sheet does not exist, create it
                with pd.ExcelWriter(excel_output_path, engine='openpyxl', mode='a') as writer:
                    qualifier_df.to_excel(
                        writer, 
                        sheet_name="Qualifying Questions", 
                        index=False
                    )
        else:
            # Create new Excel file with "Qualifying Questions" sheet
            with pd.ExcelWriter(excel_output_path, mode='w', engine='openpyxl') as writer:
                qualifier_df.to_excel(
                    writer, 
                    sheet_name="Qualifying Questions", 
                    index=False
                )
        logging.info("Qualifier checks completed and saved to Excel.")
    except Exception as e:
        logging.error("Error saving results to Excel: %s", e)
        return

    # Open the workbook for formatting
    try:
        wb = load_workbook(excel_output_path)
        ws = wb["Qualifying Questions"]

        # Determine the start row for data (assuming headers are in the first row)
        data_start_row = 2

        # Define color fills
        status_fill_pass = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
        status_fill_fail = PatternFill(start_color="FF7F7F", end_color="FF7F7F", fill_type="solid")  # Light red

        # Apply color coding to Status column (Column B)
        for row in range(data_start_row, ws.max_row + 1):
            status_cell = ws.cell(row=row, column=2)  # Column B
            if status_cell.value == "Pass":
                status_cell.fill = status_fill_pass
            elif status_cell.value == "Fail":
                status_cell.fill = status_fill_fail

        # Determine overall SOC viability
        statuses = [ws.cell(row=row, column=2).value for row in range(data_start_row, ws.max_row + 1)]
        overall_viability = "Pass" if all(status == "Pass" for status in statuses) else "Fail"

        # Define summary row data
        summary_question = "Overall SOC Viability"
        summary_status = overall_viability
        summary_answer = "SOC is valid." if overall_viability == "Pass" else "SOC is not valid."

        # Append summary row
        ws.append([summary_question, summary_status, summary_answer])

        # Get the summary row number
        summary_row = ws.max_row

        # Apply color coding to summary Status cell
        summary_fill = status_fill_pass if summary_status == "Pass" else status_fill_fail
        ws.cell(row=summary_row, column=2).fill = summary_fill

        # Make the summary row bold
        bold_font = Font(bold=True)
        for col in range(1, 4):
            ws.cell(row=summary_row, column=col).font = bold_font

        # Save the workbook
        wb.save(excel_output_path)
        logging.info("Excel formatting and summary row added.")
    except Exception as e:
        logging.error("Error formatting Excel file: %s", e)
        return

def evaluate_soc_report_minimal(df_chunks_path, faiss_index_path, top_k=3):
    """
    Minimal version of qualifier checks that:
      1) Loads the chunked text and FAISS index for qualifiers.
      2) Calls the three checks (is_report_latest, are_trust_principles_covered, is_audit_period_sufficient).
      3) Returns a simple dictionary with pass/fail (True/False) for each check + an 'overall' key.

    DOES NOT write anything to Excel (used only to see if SOC is valid enough to continue).
    """

    try:
        # 1. Load model, data, and index
        model = SentenceTransformer('all-mpnet-base-v2')
        df_chunks = pd.read_csv(df_chunks_path)
        index = load_faiss_index(faiss_index_path)

        # 2. Perform the checks
        latest_result = is_report_latest(df_chunks, model, index, top_k=top_k)
        trust_principles_result = are_trust_principles_covered(df_chunks, model, index, top_k=top_k)
        audit_period_result = is_audit_period_sufficient(df_chunks, model, index, top_k=top_k)

        # 3. Parse each result to a boolean pass/fail
        latest_pass = latest_result.strip().lower().startswith("yes.")
        trust_pass = trust_principles_result.strip().lower().startswith("yes.")
        audit_pass = audit_period_result.strip().lower().startswith("yes.")

        overall = (latest_pass and trust_pass and audit_pass)

        return {
            "latest": latest_pass,
            "trust_principles": trust_pass,
            "audit_period": audit_pass,
            "overall": overall
        }

    except Exception as e:
        logging.error(f"Error in minimal qualifier checks: {e}")
        # If error, treat as fail or return a safe default
        return {
            "latest": False,
            "trust_principles": False,
            "audit_period": False,
            "overall": False
        }
