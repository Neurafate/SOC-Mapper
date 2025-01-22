import os
import logging
import threading
import time
import json
import uuid
import re
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from parser import (
    generate_regex_from_sample,
    extract_text_from_pdf,
    chunk_text_by_multiple_patterns,
)
from rag import build_rag_system_with_parser, process_cybersecurity_framework_with_rag
from llm_analysis import (
    load_responses,
    process_controls,
    load_excel_file,
    map_columns_by_position,
    merge_dataframes,
    create_final_dataframe,
    save_to_excel,
    remove_not_met_controls
)
from qualifiers import qualify_soc_report

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --------------------------------------------------------
# Configuration – updated paths to /Work/SOC-AI subfolders
# --------------------------------------------------------
app.config['UPLOAD_FOLDER'] = '/Work/SOC-AI/uploads'
app.config['RESULTS_FOLDER'] = '/Work/SOC-AI/results'
app.config['EXCEL_FOLDER'] = '/Work/SOC-AI/excel_outputs'
app.config['RAG_OUTPUTS'] = '/Work/SOC-AI/rag_outputs'
app.config['CHUNK_SIZE'] = 1000  # Large chunk size for better context

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXCEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['RAG_OUTPUTS'], exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='LLM_analysis.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Application started.")

# Dictionary to track progress and final result for each task_id
progress_data = {}


def count_controls(excel_path):
    """
    Counts the number of controls (rows) in the provided Excel file.
    Supports both .xlsx and .csv formats.
    """
    try:
        logging.info(f"Counting controls in {excel_path}.")
        if excel_path.endswith('.xlsx') or excel_path.endswith('.xls'):
            df = pd.read_excel(excel_path)
        elif excel_path.endswith('.csv'):
            df = pd.read_csv(excel_path)
        else:
            raise ValueError("Unsupported file format for counting controls.")
        num_controls = df.shape[0]
        logging.info(f"Number of controls counted: {num_controls}")
        return num_controls
    except Exception as e:
        logging.error(f"Error counting controls in {excel_path}: {e}", exc_info=True)
        return 0


def format_qualifier_sheet(excel_path):
    """Format the Qualifying Questions sheet in the Excel output."""
    try:
        logging.info(f"Formatting Qualifying Questions sheet in {excel_path}.")
        wb = load_workbook(excel_path)
        if "Qualifying Questions" in wb.sheetnames:
            ws = wb["Qualifying Questions"]

            # Set column width and wrap text
            for col in ws.columns:
                for cell in col:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
            ws.column_dimensions['A'].width = 40
            ws.column_dimensions['B'].width = 40

            # Apply header styling
            header_fill = PatternFill(start_color="FFCCFF", end_color="FFCCFF", fill_type="solid")
            bold_font = Font(bold=True)
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = bold_font

            # Apply sentiment-based color coding
            for row in ws.iter_rows(min_row=2):
                answer_cell = row[1]  # Assuming Answer column is second
                if "yes" in str(answer_cell.value).lower():
                    answer_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Green
                elif "no" in str(answer_cell.value).lower():
                    answer_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # Red

            wb.save(excel_path)
            logging.info("Qualifying Questions sheet formatted successfully.")
        else:
            logging.warning("Qualifying Questions sheet not found in the Excel file.")
    except Exception as e:
        logging.error(f"Error formatting Qualifying Questions sheet: {e}", exc_info=True)


def chunk_text_without_patterns(text, chunk_size):
    """
    Splits the text into chunks of specified size without considering any patterns.
    Ensures that chunks do not break sentences abruptly.
    """
    try:
        logging.info("Starting text chunking without patterns.")
        text = text.replace('\n', ' ').replace('\r', ' ')
        sentences = re.split('(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            current_length = len(current_chunk) + len(sentence) + 1
            if current_length <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        logging.info(f"Text chunked into {len(chunks)} parts without patterns.")
        return chunks
    except Exception as e:
        logging.error(f"Error in chunk_text_without_patterns: {e}", exc_info=True)
        return []


def format_eta(seconds):
    """
    Converts seconds to a string in the format 'Xm Ys'.
    """
    minutes = seconds // 60
    secs = seconds % 60
    return f"{int(minutes)}m {int(secs)}s"


def sleep_seconds(task_id, seconds):
    """
    Sleeps for the specified number of seconds, decrementing the ETA each second.
    Checks for cancellation after each second.
    """
    for _ in range(seconds):
        time.sleep(1)
        if task_id in progress_data:
            # Check if cancelled
            if progress_data[task_id].get('cancelled'):
                logging.info(f"Task {task_id} has been cancelled during sleep.")
                break
            if progress_data[task_id]['eta'] > 0:
                progress_data[task_id]['eta'] -= 1


def rename_sheet_to_soc_mapping(excel_path):
    try:
        logging.info(f"Renaming 'Sheet1' to 'Control Assessment' in {excel_path}.")
        wb = load_workbook(excel_path)
        if "Sheet1" in wb.sheetnames:
            ws = wb["Sheet1"]
            ws.title = "Control Assessment"
            wb.save(excel_path)
            logging.info(f"Renamed 'Sheet1' to 'Control Assessment' in {excel_path}.")
        else:
            logging.warning("Sheet1 not found in the Excel file.")
    except Exception as e:
        logging.error(f"Error renaming Sheet1 to Control Assessment: {e}", exc_info=True)


def format_compliance_sheet(excel_path):
    """
    Format the Compliance Score sheet as requested:
    - Columns: Domain Name, Domain Compliance Score, Sub-Domain Name, Sub-Domain Compliance Score, Overall Compliance Score
    - Merge cells for each domain in Domain Name and Domain Compliance Score columns
    - Merge Overall Compliance Score top-to-bottom
    - Wrap text
    """
    try:
        logging.info(f"Formatting Compliance Score sheet in {excel_path}.")
        wb = load_workbook(excel_path)
        if "Compliance Score" not in wb.sheetnames:
            logging.warning("Compliance Score sheet not found for formatting.")
            return

        ws = wb["Compliance Score"]

        # Set wrap text for all cells
        for row in ws.iter_rows(min_row=1):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

        # Identify columns by name
        domain_col = None
        domain_score_col = None
        subdomain_col = None
        subdomain_score_col = None
        overall_col = None

        for i, cell in enumerate(ws[1], start=1):
            if cell.value == "Domain Name":
                domain_col = i
            elif cell.value == "Domain Compliance Score":
                domain_score_col = i
            elif cell.value == "Sub-Domain Name":
                subdomain_col = i
            elif cell.value == "Sub-Domain Compliance Score":
                subdomain_score_col = i
            elif cell.value == "Overall Compliance Score":
                overall_col = i

        if not all([domain_col, domain_score_col, subdomain_col, subdomain_score_col, overall_col]):
            logging.error("One or more required columns not found in Compliance Score sheet.")
            return

        # Get last row
        max_row = ws.max_row

        # Merge the Overall Compliance Score column top to bottom (from row 2 to max_row)
        if max_row > 1:
            ws.merge_cells(start_row=2, start_column=overall_col, end_row=max_row, end_column=overall_col)

        # Group rows by domain to merge domain name & domain compliance score
        current_domain = None
        domain_start_row = 2

        for row in range(2, max_row + 1):
            cell_value = ws.cell(row=row, column=domain_col).value
            if cell_value != current_domain:
                if current_domain is not None:
                    # Merge domain name
                    ws.merge_cells(
                        start_row=domain_start_row,
                        start_column=domain_col,
                        end_row=row - 1,
                        end_column=domain_col
                    )
                    # Merge domain compliance score
                    ws.merge_cells(
                        start_row=domain_start_row,
                        start_column=domain_score_col,
                        end_row=row - 1,
                        end_column=domain_score_col
                    )
                current_domain = cell_value
                domain_start_row = row
        # Merge last block
        if current_domain is not None and domain_start_row < max_row:
            ws.merge_cells(
                start_row=domain_start_row,
                start_column=domain_col,
                end_row=max_row,
                end_column=domain_col
            )
            ws.merge_cells(
                start_row=domain_start_row,
                start_column=domain_score_col,
                end_row=max_row,
                end_column=domain_score_col
            )

        # Adjust column widths
        ws.column_dimensions[get_column_letter(domain_col)].width = 25
        ws.column_dimensions[get_column_letter(domain_score_col)].width = 25
        ws.column_dimensions[get_column_letter(subdomain_col)].width = 25
        ws.column_dimensions[get_column_letter(subdomain_score_col)].width = 25
        ws.column_dimensions[get_column_letter(overall_col)].width = 25

        # Header styling
        header_fill = PatternFill(start_color="FFCCFF", end_color="FFCCFF", fill_type="solid")
        bold_font = Font(bold=True)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = bold_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

        wb.save(excel_path)
        logging.info("Compliance Score sheet formatted successfully.")
    except Exception as e:
        logging.error(f"Error formatting Compliance Score sheet: {e}", exc_info=True)


def create_executive_summary(input_excel_path, output_excel_path):
    """
    Creates an Executive Summary sheet inside the final Excel workbook using openpyxl.
    The summary includes:
      - Overall Compliance Score (average of Domain-wise Compliance Scores)
      - Domain-wise Compliance Scores
    """
    try:
        # Read the "Control Assessment" sheet using pandas for quick math
        df = pd.read_excel(input_excel_path, sheet_name='Control Assessment')

        # Check required columns
        required_cols = ["User Org Control Domain", "Compliance Score"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in 'Control Assessment'.")

        # Compute domain-wise compliance scores
        domain_avg = df.groupby("User Org Control Domain")["Compliance Score"].mean().reset_index()
        domain_avg["Compliance Score"] = domain_avg["Compliance Score"] / 100.0

        # Compute overall compliance score as the mean of domain-wise averages
        overall_avg = domain_avg["Compliance Score"].mean()

        # Load with openpyxl
        wb = load_workbook(input_excel_path)

        # If an 'Executive Summary' sheet exists, remove it
        if "Executive Summary" in wb.sheetnames:
            del wb["Executive Summary"]

        # Create a new sheet for the summary
        ws = wb.create_sheet("Executive Summary")

        # Insert Executive Summary as the first sheet
        wb._sheets.insert(0, wb._sheets.pop(wb.sheetnames.index("Executive Summary")))

        # Write headers/info
        ws["B2"] = "Overall Compliance Score"
        ws["C2"] = overall_avg
        ws["C2"].number_format = "0.00%"

        ws["B3"] = "Domain-wise Compliance Score"
        ws["C3"] = "Average Compliance Score per Domain"

        # Style the headers (B2:C3)
        header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        for row in range(2, 4):
            for col in range(2, 4):
                cell = ws.cell(row=row, column=col)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

        # Write domain-wise compliance starting from row 4
        start_row = 4
        for i, row_data in domain_avg.iterrows():
            ws.cell(row=start_row + i, column=2, value=row_data["User Org Control Domain"])
            ws.cell(row=start_row + i, column=3, value=row_data["Compliance Score"])
            ws.cell(row=start_row + i, column=3).number_format = "0.00%"

        # Set row height for Overall Compliance Score
        ws.row_dimensions[2].height = 50

        # Add border to all used cells
        thin_border = Border(
            left=Side(border_style="thin"),
            right=Side(border_style="thin"),
            top=Side(border_style="thin"),
            bottom=Side(border_style="thin")
        )

        for row in ws.iter_rows(min_row=2, max_row=start_row + len(domain_avg) - 1, min_col=2, max_col=3):
            for cell in row:
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

        # Autofit columns B and C
        ws.column_dimensions['B'].width = 40
        ws.column_dimensions['C'].width = 30

        # Save as a new file
        wb.save(output_excel_path)
        wb.close()

        logging.info(f"Executive Summary created and saved to {output_excel_path}.")

    except Exception as e:
        logging.error(f"Error in create_executive_summary: {e}", exc_info=True)
        raise


def background_process(task_id, pdf_path, excel_path, start_page, end_page, control_id, model_name, soc_report_filename, framework_filename):
    """
    The long-running background process that extracts PDF text,
    builds a RAG system, processes the controls with LLM, merges
    the analysis with the user-provided framework, performs qualifiers,
    and finally creates an Executive Summary in the final Excel file.
    """
    try:
        logging.info(f"Background processing started for task_id: {task_id}.")
        # Validate page numbers
        if start_page < 1 or end_page < start_page:
            raise ValueError("Invalid page range: ensure that 1 <= start_page <= end_page.")

        # Adjust qualifiers' page range
        qualifiers_start_page = 1
        qualifiers_end_page = start_page - 1 if start_page > 1 else 1

        # Timings
        pre_llm_time = 20
        pre_llm_steps = 6
        pre_llm_step_time = int(pre_llm_time / pre_llm_steps)
        
        # Adjust qualifier_time based on model_name
        if model_name.lower() == 'phi4':
            qualifier_time = 165
            logging.info(f"Model '{model_name}' selected. Setting qualifier_time to {qualifier_time} seconds.")
        else:
            qualifier_time = 30
            logging.info(f"Model '{model_name}' selected. Setting qualifier_time to {qualifier_time} seconds.")

        qualifier_time = max(qualifier_time, 0)  # Ensure non-negative

        # Removed compliance_scoring_time as per request
        llm_time_per_control = 55 if model_name.lower() == 'phi4' else 15

        num_controls = count_controls(excel_path)
        llm_time = num_controls * llm_time_per_control

        # Total ETA includes pre_llm_time, llm_time, and qualifier_time
        total_eta = pre_llm_time + llm_time + qualifier_time
        progress_data[task_id]['eta'] = int(total_eta)
        progress_data[task_id]['progress'] = 0.0
        progress_data[task_id]['status'] = "Initializing..."
        progress_data[task_id]['download_url'] = None
        progress_data[task_id]['error'] = None
        progress_data[task_id]['num_controls'] = num_controls

        progress_increment_pre_llm = 20.0 / pre_llm_steps
        progress_increment_llm = 70.0 / num_controls if num_controls > 0 else 0
        progress_increment_qualifier = 10.0

        # Helper function to check for cancellation
        def check_cancel(tid):
            if progress_data[tid].get('cancelled'):
                logging.info(f"Task {tid} has been cancelled.")
                progress_data[tid]['status'] = "Cancelled"
                progress_data[tid]['progress'] = 0
                progress_data[tid]['eta'] = 0
                progress_data[tid]['error'] = "Task was cancelled by the user."
                raise Exception("Task cancelled by user.")

        # -------------- Pre-LLM phases --------------
        pre_llm_phase_descriptions = [
            "Extracting full text for qualifiers...",
            "Chunking full text for qualifiers...",
            "Building RAG system for qualifiers...",
            "Extracting controls' text...",
            "Chunking controls' text...",
            "Building RAG system for controls..."
        ]

        for idx, step_description in enumerate(pre_llm_phase_descriptions):
            check_cancel(task_id)
            progress_data[task_id]['status'] = step_description
            logging.info(f"Task {task_id}: {step_description}, ETA: {format_eta(progress_data[task_id]['eta'])}")

            if idx == 0:
                # 1: Extracting full text for qualifiers
                full_text_output_path = os.path.join(app.config['RAG_OUTPUTS'], f"full_text_{task_id}.txt")
                full_extracted_text = extract_text_from_pdf(pdf_path, qualifiers_start_page, qualifiers_end_page, full_text_output_path)
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(full_text_output_path):
                    raise FileNotFoundError(f"Extracted full text file not found at {full_text_output_path}")
                progress_data[task_id]['progress'] += progress_increment_pre_llm

            elif idx == 1:
                # 2: Chunking full text for qualifiers
                chunk_size = app.config['CHUNK_SIZE']
                full_text_chunks = chunk_text_without_patterns(full_extracted_text, chunk_size)
                df_qualifier_chunks_path = os.path.join(app.config['RAG_OUTPUTS'], f"df_qualifier_chunks_{task_id}.csv")
                pd.DataFrame({"Content": full_text_chunks}).to_csv(df_qualifier_chunks_path, index=False)
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(df_qualifier_chunks_path):
                    raise FileNotFoundError(f"Qualifier chunks file not found at {df_qualifier_chunks_path}")
                progress_data[task_id]['progress'] += progress_increment_pre_llm

            elif idx == 2:
                # 3: Building RAG system for qualifiers
                faiss_index_qualifiers_path = os.path.join(app.config['RAG_OUTPUTS'], f"faiss_index_qualifiers_{task_id}.idx")
                build_rag_system_with_parser(
                    pdf_path=pdf_path,
                    start_page=qualifiers_start_page,
                    end_page=qualifiers_end_page,
                    control_patterns=None,
                    output_text_path=full_text_output_path,
                    df_chunks_path=df_qualifier_chunks_path,
                    faiss_index_path=faiss_index_qualifiers_path,
                    chunk_size=chunk_size
                )
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(faiss_index_qualifiers_path):
                    raise FileNotFoundError(f"Qualifiers FAISS index not found at {faiss_index_qualifiers_path}")
                progress_data[task_id]['progress'] += progress_increment_pre_llm

            elif idx == 3:
                # 4: Extracting controls' text
                controls_output_text_path = os.path.join(app.config['RAG_OUTPUTS'], f"controls_text_{task_id}.txt")
                controls_extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page, controls_output_text_path)
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(controls_output_text_path):
                    raise FileNotFoundError(f"Controls extracted text file not found at {controls_output_text_path}")
                progress_data[task_id]['progress'] += progress_increment_pre_llm

            elif idx == 4:
                # 5: Chunking controls' text
                control_ids_raw = control_id
                control_ids = [cid.strip() for cid in control_ids_raw.split(',') if cid.strip()]
                if not control_ids:
                    raise ValueError("No valid Control IDs provided.")

                # Generate regex patterns
                control_patterns = [generate_regex_from_sample(cid) for cid in control_ids]
                logging.info(f"Generated regex patterns: {control_patterns}")

                # Chunk text using the generated regex patterns
                control_text_chunks = chunk_text_by_multiple_patterns(controls_extracted_text, control_patterns)
                df_control_chunks_path = os.path.join(app.config['RAG_OUTPUTS'], f"df_control_chunks_{task_id}.csv")
                control_text_chunks.to_csv(df_control_chunks_path, index=False)
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(df_control_chunks_path):
                    raise FileNotFoundError(f"Control chunks file not found at {df_control_chunks_path}")
                progress_data[task_id]['progress'] += progress_increment_pre_llm

            elif idx == 5:
                # 6: Building RAG system for controls
                faiss_index_controls_path = os.path.join(app.config['RAG_OUTPUTS'], f"faiss_index_controls_{task_id}.idx")
                build_rag_system_with_parser(
                    pdf_path=pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                    control_patterns=control_patterns,
                    output_text_path=controls_output_text_path,
                    df_chunks_path=df_control_chunks_path,
                    faiss_index_path=faiss_index_controls_path,
                    chunk_size=app.config['CHUNK_SIZE']
                )
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(faiss_index_controls_path):
                    raise FileNotFoundError(f"Controls FAISS index not found at {faiss_index_controls_path}")
                progress_data[task_id]['progress'] += progress_increment_pre_llm

        # Process control framework with RAG
        check_cancel(task_id)
        progress_data[task_id]['status'] = "Processing control framework with RAG..."
        logging.info(f"Task {task_id}: Processing control framework with RAG. ETA: {format_eta(progress_data[task_id]['eta'])}")
        processed_framework_path = os.path.join(app.config['RAG_OUTPUTS'], f"cybersecurity_framework_with_answers_{task_id}.xlsx")
        process_cybersecurity_framework_with_rag(
            excel_input_path=excel_path,
            output_path=processed_framework_path,
            faiss_index_path=faiss_index_controls_path,
            df_chunks_path=df_control_chunks_path,
            top_k=3
        )
        check_cancel(task_id)
        if not os.path.exists(processed_framework_path):
            raise FileNotFoundError(f"Processed framework file not found at {processed_framework_path}")

        # Analyze controls with LLM
        progress_data[task_id]['status'] = "Analyzing controls with LLM..."
        logging.info(f"Task {task_id}: Analyzing controls with LLM. ETA: {format_eta(progress_data[task_id]['eta'])}")
        analysis_df = load_responses(processed_framework_path)
        analyzed_rows = []

        for i, row in analysis_df.iterrows():
            check_cancel(task_id)
            progress_data[task_id]['status'] = f"Analyzing control {i+1} of {num_controls} with LLM..."
            logging.info(f"Task {task_id}: Analyzing control {i+1}/{num_controls}. ETA: {format_eta(progress_data[task_id]['eta'])}")

            # Sleep 1 second, then decrement the rest
            sleep_seconds(task_id, 1)
            if progress_data[task_id]['eta'] > (llm_time_per_control - 1):
                progress_data[task_id]['eta'] -= (llm_time_per_control - 1)

            # Process exactly one control at a time, passing the chosen model
            processed_row = process_controls(pd.DataFrame([row]), model_name=model_name)
            analyzed_rows.append(processed_row)
            progress_data[task_id]['progress'] += progress_increment_llm

        analyzed_df = pd.concat(analyzed_rows, ignore_index=True) if analyzed_rows else pd.DataFrame()

        # **Call remove_not_met_controls after LLM analysis and Explanation is populated**
        analyzed_df = remove_not_met_controls(analyzed_df)

        # Save analyzed_df to Excel
        analysis_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"Framework_Analysis_Completed_{task_id}.xlsx")
        analyzed_df.to_excel(analysis_output_path, index=False)
        logging.info(f"Task {task_id}: LLM analysis completed at {analysis_output_path}")
        check_cancel(task_id)

        # Merge and finalize
        framework_df = load_excel_file(excel_path)
        framework_df, error = map_columns_by_position(framework_df)
        if error:
            raise ValueError(error)

        merged_df = merge_dataframes(framework_df, analyzed_df)
        final_df, error = create_final_dataframe(merged_df)
        if error:
            raise ValueError(error)

        final_output_path = os.path.join(app.config['EXCEL_FOLDER'], f"Final_Control_Status_{task_id}.xlsx")
        save_to_excel(final_df, final_output_path)
        logging.info(f"Task {task_id}: Final control status saved to {final_output_path}")
        progress_data[task_id]['progress'] = 90.0
        rename_sheet_to_soc_mapping(final_output_path)
        check_cancel(task_id)


        # Qualifier checks
        progress_data[task_id]['status'] = "Performing qualifier checks..."
        logging.info(f"Task {task_id}: Performing qualifier checks. ETA: {format_eta(progress_data[task_id]['eta'])}")
        try:
            qualify_soc_report(
                pdf_path=pdf_path,
                df_chunks_path=os.path.join(app.config['RAG_OUTPUTS'], f"df_qualifier_chunks_{task_id}.csv"),
                faiss_index_path=os.path.join(app.config['RAG_OUTPUTS'], f"faiss_index_qualifiers_{task_id}.idx"),
                excel_output_path=final_output_path
            )
            logging.info(f"Task {task_id}: Qualifier checks completed.")
        except Exception as q_ex:
            logging.error(f"Error during qualifier checks: {q_ex}", exc_info=True)

        format_qualifier_sheet(final_output_path)
        
        # Replace the fixed sleep with dynamic qualifier_time
        sleep_seconds(task_id, qualifier_time)  # Adjusted sleep based on model
        check_cancel(task_id)
        progress_data[task_id]['progress'] += progress_increment_qualifier

        # === Create Executive Summary in a new final file ===
        progress_data[task_id]['status'] = "Creating Executive Summary..."
        logging.info(f"Task {task_id}: Creating Executive Summary. ETA: {format_eta(progress_data[task_id]['eta'])}")
        
        # Generate the final filename based on input filenames
        soc_report_basename = os.path.splitext(os.path.basename(soc_report_filename))[0]
        framework_basename = os.path.splitext(os.path.basename(framework_filename))[0]
        final_filename = f"{soc_report_basename}_Baselined_Vs_{framework_basename}.xlsx"
        summary_output_path = os.path.join(app.config['EXCEL_FOLDER'], final_filename)
        
        create_executive_summary(final_output_path, summary_output_path)
        logging.info(f"Task {task_id}: Executive Summary created at {summary_output_path}")

        # **Set the download URL to the final summarized Excel file**
        progress_data[task_id]['download_url'] = f"http://127.0.0.1:5000/download/{final_filename}"

        # **Update status and ETA after Executive Summary creation**
        progress_data[task_id]['status'] = "Executive Summary is being prepared..."
        # Assuming the Executive Summary preparation takes some time, but since it's already created,
        # we can immediately set progress to 100 and ETA to 0.

        progress_data[task_id]['progress'] = 100.0
        progress_data[task_id]['eta'] = 0
        progress_data[task_id]['status'] = "Task completed successfully."
        logging.info(f"Task {task_id}: Task completed successfully.")

    except Exception as e:
        logging.error(f"Error in background_process (task_id: {task_id}): {e}", exc_info=True)
        if "cancelled" in str(e).lower():
            # Mark the task as cancelled
            progress_data[task_id]['cancelled'] = True
            progress_data[task_id]['status'] = "Cancelled"
            progress_data[task_id]['progress'] = 0
            progress_data[task_id]['eta'] = 0
            progress_data[task_id]['error'] = "Task was cancelled by the user."
        else:
            progress_data[task_id]['status'] = "Error occurred."
            progress_data[task_id]['error'] = str(e)
            progress_data[task_id]['eta'] = 0


@app.route('/process_all', methods=['POST'])
def process_all():
    logging.info("Received request to /process_all endpoint.")
    try:
        pdf_file = request.files.get('pdf_file')
        excel_file = request.files.get('excel_file')

        if not pdf_file or not excel_file:
            raise ValueError("Missing required files: pdf_file or excel_file")

        try:
            start_page = int(request.form.get('start_page', 1))
            end_page = int(request.form.get('end_page', 10))
        except ValueError:
            logging.error("Invalid page numbers provided.")
            return jsonify({'error': 'start_page and end_page must be integers'}), 400

        control_id = request.form.get('control_id', '')
        if not control_id:
            raise ValueError("Missing required parameter: control_id")

        # Retrieve model name from the frontend; default to 'llama3.1' if not provided
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

        # Start background processing thread
        thread = threading.Thread(
            target=background_process, 
            args=(task_id, pdf_path, excel_path, start_page, end_page, control_id, model_name, soc_report_filename, framework_filename)
        )
        thread.start()

        return jsonify({"message": "Processing started", "task_id": task_id}), 202

    except Exception as e:
        logging.error(f"Error in /process_all endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route('/progress/<task_id>', methods=['GET'])
def sse_progress(task_id):
    """
    SSE endpoint that streams progress updates for the given task_id.
    """
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
                'eta': round(eta, 2) if eta is not None and isinstance(eta, (int, float)) else eta
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

    return Response(generate(), mimetype='text/event-stream')


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
    # Note: Consider setting debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
