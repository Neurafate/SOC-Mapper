import os
import logging
import pandas as pd
from rag import retrieve_answers_for_controls, load_faiss_index
from llm_analysis import call_ollama_api
from sentence_transformers import SentenceTransformer
import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # Save results to the Excel file
    try:
        if os.path.exists(excel_output_path):
            with pd.ExcelWriter(excel_output_path, mode='a', engine='openpyxl') as writer:
                # Check if "Qualifier Results" sheet exists
                try:
                    existing_df = pd.read_excel(excel_output_path, sheet_name="Qualifier Results")
                    # If exists, append without overwriting
                    qualifier_df = pd.DataFrame(qualifier_results)
                    startrow = existing_df.shape[0] + 1
                    qualifier_df.to_excel(
                        writer, 
                        sheet_name="Qualifying Questions", 
                        index=False, 
                        header=False, 
                        startrow=startrow
                    )
                except ValueError:
                    # If "Qualifier Results" sheet does not exist, create it
                    pd.DataFrame(qualifier_results).to_excel(
                        writer, 
                        sheet_name="Qualifying Questions", 
                        index=False
                    )
        else:
            with pd.ExcelWriter(excel_output_path, mode='w', engine='openpyxl') as writer:
                pd.DataFrame(qualifier_results).to_excel(
                    writer, 
                    sheet_name="Qualifying Questions", 
                    index=False
                )
        logging.info("Qualifier checks completed and saved to Excel.")
    except Exception as e:
        logging.error("Error saving results to Excel: %s", e)

if __name__ == "__main__":
    # Example usage (you can remove or adjust this as needed)
    pdf_path = "path/to/soc2_report.pdf"
    df_chunks_path = "path/to/df_chunks.csv"
    faiss_index_path = "path/to/faiss_index"
    excel_output_path = "path/to/output.xlsx"

    qualify_soc_report(pdf_path, df_chunks_path, faiss_index_path, excel_output_path)
