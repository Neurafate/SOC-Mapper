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
    try:
        distances, indices = index.search(query_emb, top_k)
        retrieved_answers = retrieve_answers_for_controls(
            pd.DataFrame([{'Control': query}]), model, index, df_chunks, top_k
        )
    except Exception as e:
        logging.error("Error retrieving answers for 'is_report_latest': %s", e)
        return "No. Unable to determine the publication date."

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

    try:
        response = call_ollama_api(prompt)
    except Exception as e:
        logging.error("Error calling LLM for 'is_report_latest': %s", e)
        return "No. Unable to determine the publication date."

    return response

def are_trust_principles_covered(df_chunks, model, index, top_k=3):
    """
    Checks if the SOC 2 Type 2 Report covers the three Trust Principles: Security, Availability, and Confidentiality.
    """
    query = "Does the SOC 2 Type 2 Report cover the principles of Security, Availability, and Confidentiality?"
    query_emb = model.encode([query], show_progress_bar=False).astype('float32')

    # Retrieve answers from RAG
    try:
        distances, indices = index.search(query_emb, top_k)
        retrieved_answers = retrieve_answers_for_controls(
            pd.DataFrame([{'Control': query}]), model, index, df_chunks, top_k
        )
    except Exception as e:
        logging.error("Error retrieving answers for 'are_trust_principles_covered': %s", e)
        return "No. Unable to determine coverage of Trust Principles."

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

    try:
        response = call_ollama_api(prompt)
    except Exception as e:
        logging.error("Error calling LLM for 'are_trust_principles_covered': %s", e)
        return "No. Unable to determine coverage of Trust Principles."

    return response

def is_audit_period_sufficient(df_chunks, model, index, top_k=3):
    """
    Checks if the SOC 2 Type 2 Report covers an audit period of at least 9 months.
    """
    query = "Does the SOC 2 Type 2 Report cover an audit period of at least 9 months?"
    query_emb = model.encode([query], show_progress_bar=False).astype('float32')

    # Retrieve answers from RAG
    try:
        distances, indices = index.search(query_emb, top_k)
        retrieved_answers = retrieve_answers_for_controls(
            pd.DataFrame([{'Control': query}]), model, index, df_chunks, top_k
        )
    except Exception as e:
        logging.error("Error retrieving answers for 'is_audit_period_sufficient': %s", e)
        return "No. Unable to determine the audit period."

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

    try:
        response = call_ollama_api(prompt)
    except Exception as e:
        logging.error("Error calling LLM for 'is_audit_period_sufficient': %s", e)
        return "No. Unable to determine the audit period."

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
