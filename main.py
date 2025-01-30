# main.py

import os
import logging
import threading
import time
import json
import uuid
import re
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from identifier import process_pdf as identify_control_ids

# Existing imports from your project
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

import PyPDF2  # New import for PDF processing

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
                if answer_cell.value:
                    answer_text = str(answer_cell.value).lower()
                    if "yes" in answer_text:
                        answer_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Green
                    elif "no" in answer_text:
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
      - Legend explaining compliance score colors
      - SOC Usability status
      - Overall Compliance Percentage
      - Domain-wise Compliance Breakdown
      - Sub-Domain-wise Compliance Breakdown
      - Count of fully met, partially met, and not met controls per domain and sub-domain
    """
    try:
        logging.info("Generating Executive Summary...")

        # Read the "Control Assessment" sheet using pandas
        df = pd.read_excel(input_excel_path, sheet_name='Control Assessment')

        # Debug: Print available columns to confirm naming
        logging.info(f"Columns in 'Control Assessment': {list(df.columns)}")

        # Expected column names
        expected_columns = [
            'Sr. No.',
            'User Org Control Domain',
            'User Org Control Sub-Domain',
            'User Org Control Statement',
            'Service Org Control IDs',
            'Service Org Controls',
            'Compliance Score',
            'Detailed Analysis Explanation',
            'Control Status'
        ]

        # Ensure all required columns exist
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in 'Control Assessment'.")

        # Data Cleaning: Trim whitespace and standardize capitalization in 'User Org Control Domain' and 'User Org Control Sub-Domain'
        df['User Org Control Domain'] = df['User Org Control Domain'].astype(str).str.strip().str.title()
        df['User Org Control Sub-Domain'] = df['User Org Control Sub-Domain'].astype(str).str.strip().str.title()

        # Debug: Log unique domains and sub-domains
        unique_domains = df['User Org Control Domain'].unique()
        logging.info(f"Unique Domains ({len(unique_domains)}): {unique_domains}")

        unique_subdomains = df['User Org Control Sub-Domain'].unique()
        logging.info(f"Unique Sub-Domains ({len(unique_subdomains)}): {unique_subdomains}")

        # Compute domain-wise compliance scores
        domain_avg = df.groupby("User Org Control Domain")["Compliance Score"].mean().reset_index()
        domain_avg["Compliance Score"] = domain_avg["Compliance Score"] / 100.0  # Convert to percentage

        # Compute overall compliance score as the mean of domain-wise averages
        overall_avg = domain_avg["Compliance Score"].mean()

        # Count fully met, partially met, and not met controls per domain
        domain_status_counts = df.groupby(["User Org Control Domain", "Control Status"]).size().unstack(fill_value=0).reset_index()

        # Ensure all categories exist
        for status in ["Fully Met", "Partially Met", "Not Met"]:
            if status not in domain_status_counts.columns:
                domain_status_counts[status] = 0

        # Merge compliance score and control status counts
        domain_summary = domain_avg.merge(domain_status_counts, on="User Org Control Domain", how="left")

        # Verify that all domains are included after merging
        if len(domain_summary) != len(unique_domains):
            missing_domains = set(unique_domains) - set(domain_summary['User Org Control Domain'])
            logging.warning(f"Missing Domains after merging: {missing_domains}")
            # Optionally, add missing domains with default values
            for domain in missing_domains:
                domain_summary = domain_summary.append({
                    "User Org Control Domain": domain,
                    "Compliance Score": 0.0,
                    "Fully Met": 0,
                    "Partially Met": 0,
                    "Not Met": 0
                }, ignore_index=True)

        # Compute sub-domain-wise compliance scores
        subdomain_avg = df.groupby("User Org Control Sub-Domain")["Compliance Score"].mean().reset_index()
        subdomain_avg["Compliance Score"] = subdomain_avg["Compliance Score"] / 100.0  # Convert to percentage

        # Count fully met, partially met, and not met controls per sub-domain
        subdomain_status_counts = df.groupby(["User Org Control Sub-Domain", "Control Status"]).size().unstack(fill_value=0).reset_index()

        # Ensure all categories exist
        for status in ["Fully Met", "Partially Met", "Not Met"]:
            if status not in subdomain_status_counts.columns:
                subdomain_status_counts[status] = 0

        # Merge compliance score and control status counts for sub-domains
        subdomain_summary = subdomain_avg.merge(subdomain_status_counts, on="User Org Control Sub-Domain", how="left")

        # Verify that all sub-domains are included after merging
        if len(subdomain_summary) != len(unique_subdomains):
            missing_subdomains = set(unique_subdomains) - set(subdomain_summary['User Org Control Sub-Domain'])
            logging.warning(f"Missing Sub-Domains after merging: {missing_subdomains}")
            # Optionally, add missing sub-domains with default values
            for subdomain in missing_subdomains:
                subdomain_summary = subdomain_summary.append({
                    "User Org Control Sub-Domain": subdomain,
                    "Compliance Score": 0.0,
                    "Fully Met": 0,
                    "Partially Met": 0,
                    "Not Met": 0
                }, ignore_index=True)

        # Load workbook with openpyxl
        wb = load_workbook(input_excel_path)

        # If an 'Executive Summary' sheet exists, remove it
        if "Executive Summary" in wb.sheetnames:
            del wb["Executive Summary"]

        # Create a new sheet for the summary
        ws = wb.create_sheet("Executive Summary")

        # Insert Executive Summary as the first sheet
        wb._sheets.insert(0, wb._sheets.pop(wb.sheetnames.index("Executive Summary")))

        # Define styles using good color theory principles
        # Use colorblind-friendly palette and ensure good contrast
        legend_colors = {
            ">90%": "CCFFCC",         # Very Light Green
            "<90% & >75%": "FFF2CC",  # Very Light Amber/Orange
            "<75%": "FFCCCC"          # Very Light Red
        }

        fill_legend = {
            ">90%": PatternFill(start_color=legend_colors[">90%"], end_color=legend_colors[">90%"], fill_type="solid"),
            "<90% & >75%": PatternFill(start_color=legend_colors["<90% & >75%"], end_color=legend_colors["<90% & >75%"], fill_type="solid"),
            "<75%": PatternFill(start_color=legend_colors["<75%"], end_color=legend_colors["<75%"], fill_type="solid")
        }

        # Use the same colors for compliance_fill_colors to ensure consistency
        compliance_fill_colors = {
            "high": legend_colors[">90%"],            # Very Light Green
            "medium": legend_colors["<90% & >75%"],  # Very Light Amber/Orange
            "low": legend_colors["<75%"]             # Very Light Red
        }

        # **Updated Header Fill Color to Gold (#FFD700)**
        header_fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")  # Gold
        header_font = Font(bold=True, color="FFFFFF")
        thin_border = Border(
            left=Side(border_style="thin"),
            right=Side(border_style="thin"),
            top=Side(border_style="thin"),
            bottom=Side(border_style="thin")
        )
        center_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        left_alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)

        # 1. Legend Section
        # Merge B2:C2 and set "Legend"
        ws.merge_cells('B2:C2')
        ws['B2'] = "Legend"
        ws['B2'].font = Font(bold=True, size=14, color="FFFFFF")
        ws['B2'].alignment = center_alignment
        ws['B2'].fill = header_fill

        # Define legend items
        legend_items = [
            { "text": ">90%", "fill": fill_legend[">90%"] },
            { "text": "<90% & >75%", "fill": fill_legend["<90% & >75%"] },
            { "text": "<75%", "fill": fill_legend["<75%"] }
        ]

        # Write legend items starting from B3
        for idx, item in enumerate(legend_items, start=3):
            ws[f'B{idx}'] = item["text"]
            ws[f'B{idx}'].alignment = left_alignment
            ws[f'B{idx}'].font = Font(bold=False, color="000000")
            ws[f'C{idx}'] = ""  # Empty cell to apply color
            ws[f'C{idx}'].fill = item["fill"]
            ws[f'C{idx}'].border = thin_border

        # Apply borders to legend section
        for row in range(2, 6):
            for col in ['B', 'C']:
                ws[f'{col}{row}'].border = thin_border

        # 2. SOC Usability Section
        ws['B7'] = "SOC Usability"
        ws['B7'].font = Font(bold=True, color="FFFFFF")
        ws['B7'].alignment = left_alignment
        ws['B7'].fill = header_fill
        ws['B7'].border = thin_border

        # **Dynamic SOC Usability Status**
        # Replace the following line with dynamic data retrieval if available
        soc_usability_status = "Yes"  # Example: "Yes" or "No"

        ws['C7'] = soc_usability_status  # This should be set dynamically based on your data
        ws['C7'].number_format = "@"  # Text format to ensure proper display
        ws['C7'].alignment = center_alignment
        ws['C7'].border = thin_border

        # **Apply Color Coding to SOC Usability Status**
        soc_usability_colors = {
            "yes": "CCFFCC",  # Very Light Green
            "no": "FFCCCC"    # Very Light Red
        }

        # Retrieve the SOC Usability value and apply corresponding fill
        soc_usability_value = ws['C7'].value
        if soc_usability_value:
            soc_usability_key = soc_usability_value.strip().lower()
            fill_color_soc = soc_usability_colors.get(soc_usability_key, None)
            if fill_color_soc:
                ws['C7'].fill = PatternFill(start_color=fill_color_soc, end_color=fill_color_soc, fill_type="solid")
            else:
                # Optional: Apply no fill or a default fill if value is neither "Yes" nor "No"
                pass

        # 3. Overall Compliance Percentage
        ws['B10'] = "Overall Compliance Percentage"
        ws['B10'].font = Font(bold=True, color="FFFFFF")
        ws['B10'].alignment = left_alignment
        ws['B10'].fill = header_fill
        ws['B10'].border = thin_border

        ws['C10'] = overall_avg
        ws['C10'].number_format = "0.00%"
        ws['C10'].alignment = center_alignment
        ws['C10'].border = thin_border

        # Color code the Overall Compliance Percentage based on the legend
        compliance_percentage_overall = overall_avg * 100  # Convert to percentage
        if compliance_percentage_overall >= 90:
            fill_color_overall = compliance_fill_colors["high"]  # Very Light Green
        elif 75 <= compliance_percentage_overall < 90:
            fill_color_overall = compliance_fill_colors["medium"]  # Very Light Amber/Orange
        else:
            fill_color_overall = compliance_fill_colors["low"]  # Very Light Red

        ws['C10'].fill = PatternFill(start_color=fill_color_overall, end_color=fill_color_overall, fill_type="solid")

        # 4. Domain-wise Compliance Breakdown Headers
        ws['B12'] = "Domain-wise Compliance Breakdown"
        ws['B12'].font = Font(bold=True, size=12, color="FFFFFF")
        ws['B12'].alignment = left_alignment
        ws['B12'].fill = header_fill
        ws['B12'].border = thin_border

        headers_domain = ["Average Compliance Score Per Domain", "Fully Met Controls", "Partially Met Controls", "Not Met Controls"]
        start_col = 3  # Column C
        for idx, header in enumerate(headers_domain, start=start_col):
            cell = ws.cell(row=12, column=idx, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_alignment
            cell.border = thin_border

        # 5. Write Domain-wise Compliance Data starting from Row 13
        start_row_domain = 13
        for i, row_data in domain_summary.iterrows():
            row_num = start_row_domain + i
            ws.cell(row=row_num, column=2, value=row_data["User Org Control Domain"])
            ws.cell(row=row_num, column=2).alignment = left_alignment
            ws.cell(row=row_num, column=2).border = thin_border

            ws.cell(row=row_num, column=3, value=row_data["Compliance Score"])
            ws.cell(row=row_num, column=3).number_format = "0.00%"
            ws.cell(row=row_num, column=3).alignment = center_alignment
            ws.cell(row=row_num, column=3).border = thin_border

            ws.cell(row=row_num, column=4, value=row_data.get("Fully Met", 0))
            ws.cell(row=row_num, column=4).alignment = center_alignment
            ws.cell(row=row_num, column=4).border = thin_border

            ws.cell(row=row_num, column=5, value=row_data.get("Partially Met", 0))
            ws.cell(row=row_num, column=5).alignment = center_alignment
            ws.cell(row=row_num, column=5).border = thin_border

            ws.cell(row=row_num, column=6, value=row_data.get("Not Met", 0))
            ws.cell(row=row_num, column=6).alignment = center_alignment
            ws.cell(row=row_num, column=6).border = thin_border

            # Apply color fill only to the "Average Compliance Score Per Domain" cell based on the legend
            compliance_percentage = row_data["Compliance Score"] * 100  # Convert back to percentage
            if compliance_percentage >= 90:
                fill_color = compliance_fill_colors["high"]  # Very Light Green
            elif 75 <= compliance_percentage < 90:
                fill_color = compliance_fill_colors["medium"]  # Very Light Amber/Orange
            else:
                fill_color = compliance_fill_colors["low"]  # Very Light Red

            ws.cell(row=row_num, column=3).fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")

        # Determine the starting row for Sub-Domain-wise Compliance Breakdown
        last_domain_row = start_row_domain + len(domain_summary) - 1
        start_row_subdomain_header = last_domain_row + 2  # One empty row

        # 6. Sub-Domain-wise Compliance Breakdown Headers
        ws.cell(row=start_row_subdomain_header, column=2, value="Sub-Domain-wise Compliance Breakdown")
        ws.cell(row=start_row_subdomain_header, column=2).font = Font(bold=True, size=12, color="FFFFFF")
        ws.cell(row=start_row_subdomain_header, column=2).alignment = left_alignment
        ws.cell(row=start_row_subdomain_header, column=2).fill = header_fill
        ws.cell(row=start_row_subdomain_header, column=2).border = thin_border

        headers_subdomain = ["Average Compliance Score Per Sub-Domain", "Fully Met Controls", "Partially Met Controls", "Not Met Controls"]
        start_col_sub = 3  # Column C
        for idx, header in enumerate(headers_subdomain, start=start_col_sub):
            cell = ws.cell(row=start_row_subdomain_header, column=idx, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_alignment
            cell.border = thin_border

        # 7. Write Sub-Domain-wise Compliance Data starting from Row 16 (Dynamic)
        start_row_subdomain = start_row_subdomain_header + 1
        for i, row_data in subdomain_summary.iterrows():
            row_num = start_row_subdomain + i
            ws.cell(row=row_num, column=2, value=row_data["User Org Control Sub-Domain"])
            ws.cell(row=row_num, column=2).alignment = left_alignment
            ws.cell(row=row_num, column=2).border = thin_border

            ws.cell(row=row_num, column=3, value=row_data["Compliance Score"])
            ws.cell(row=row_num, column=3).number_format = "0.00%"
            ws.cell(row=row_num, column=3).alignment = center_alignment
            ws.cell(row=row_num, column=3).border = thin_border

            ws.cell(row=row_num, column=4, value=row_data.get("Fully Met", 0))
            ws.cell(row=row_num, column=4).alignment = center_alignment
            ws.cell(row=row_num, column=4).border = thin_border

            ws.cell(row=row_num, column=5, value=row_data.get("Partially Met", 0))
            ws.cell(row=row_num, column=5).alignment = center_alignment
            ws.cell(row=row_num, column=5).border = thin_border

            ws.cell(row=row_num, column=6, value=row_data.get("Not Met", 0))
            ws.cell(row=row_num, column=6).alignment = center_alignment
            ws.cell(row=row_num, column=6).border = thin_border

            # Apply color fill only to the "Average Compliance Score Per Sub-Domain" cell based on the legend
            compliance_percentage = row_data["Compliance Score"] * 100  # Convert back to percentage
            if compliance_percentage >= 90:
                fill_color = compliance_fill_colors["high"]  # Very Light Green
            elif 75 <= compliance_percentage < 90:
                fill_color = compliance_fill_colors["medium"]  # Very Light Amber/Orange
            else:
                fill_color = compliance_fill_colors["low"]  # Very Light Red

            ws.cell(row=row_num, column=3).fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")

        # 8. Optimize Column Widths Based on Content
        def autofit_columns(ws, min_width=10, max_width=50):
            """
            Adjust column widths based on the maximum length of the content in each column.
            """
            for column_cells in ws.columns:
                length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
                adjusted_width = length + 2  # Adding some padding
                adjusted_width = max(min_width, min(adjusted_width, max_width))
                column_letter = get_column_letter(column_cells[0].column)
                ws.column_dimensions[column_letter].width = adjusted_width

        autofit_columns(ws)

        # 9. Ensure Text Wrapping for All Cells
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=2, max_col=6):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

        # 10. Save as a new file
        wb.save(output_excel_path)
        wb.close()

        logging.info(f"Executive Summary created and saved to {output_excel_path}.")

    except Exception as e:
        logging.error(f"Error in create_executive_summary: {e}", exc_info=True)
        raise

def detect_control_id_pages(pdf_path, regex_to_cids):
    """
    Detects the pages where each Control ID appears in the PDF.
    Groups Control IDs by their regex patterns and assigns the earliest page
    where any Control ID in the group appears as the start page for all in the group.

    :param pdf_path: Path to the PDF file.
    :param regex_to_cids: Dict mapping regex patterns to list of Control IDs.
    :return: Dictionary mapping Control IDs to list of page numbers where they appear.
    """
    control_id_pages = {cid: [] for cids in regex_to_cids.values() for cid in cids}
    try:
        logging.info(f"Detecting pages for Control IDs in {pdf_path}.")
        reader = PyPDF2.PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                for regex, cids in regex_to_cids.items():
                    if re.search(regex, text, re.IGNORECASE):
                        for cid in cids:
                            if page_num not in control_id_pages[cid]:
                                control_id_pages[cid].append(page_num)
                                logging.debug(f"Found Control ID '{cid}' on page {page_num}.")
        logging.info(f"Control ID pages detected: {control_id_pages}")
        return control_id_pages
    except Exception as e:
        logging.error(f"Error in detect_control_id_pages: {e}", exc_info=True)
        return control_id_pages  # Return what has been found so far

def determine_page_range(control_id_pages, regex_to_cids):
    """
    Determines the start and end pages based on Control IDs' page occurrences, 
    prioritizing the order of regex patterns.

    :param control_id_pages: Dictionary mapping Control IDs to list of page numbers.
    :param regex_to_cids: Dict mapping regex patterns to list of Control IDs.
    :return: Tuple (start_page, end_page)
    """
    start_pages = []
    for regex, cids in regex_to_cids.items():
        # Find the earliest page among all Control IDs in this regex group
        pages = [min(control_id_pages[cid]) for cid in cids if control_id_pages.get(cid)]
        if pages:
            earliest_page = min(pages)
            start_pages.append(earliest_page)
            logging.info(f"Earliest page for regex '{regex}': {earliest_page}")

    # The overall start_page is the minimum of all earliest_pages
    start_page = min(start_pages) if start_pages else 1
    logging.info(f"Overall start_page determined: {start_page}")

    # The end_page is the maximum page across all Control IDs
    all_pages = [p for pages in control_id_pages.values() for p in pages]
    end_page = max(all_pages) if all_pages else 1
    logging.info(f"Overall end_page determined: {end_page}")

    return (start_page, end_page)

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
        pre_llm_time = 40  # Updated total time for pre-LLM steps in seconds
        pre_llm_steps = 6
        pre_llm_step_time = int(pre_llm_time / pre_llm_steps)  # Approximately 6.67 seconds per step
        
        # Determine qualifier_time based on model_name
        if model_name.lower() == 'phi4':
            qualifier_time = 21  # Qualifier processing time for Phi4
            logging.info(f"Model '{model_name}' selected. Setting qualifier_time to {qualifier_time} seconds.")
        else:
            qualifier_time = 9   # Qualifier processing time for Llama3.1
            logging.info(f"Model '{model_name}' selected. Setting qualifier_time to {qualifier_time} seconds.")
        
        # Determine LLM processing time per control
        llm_time_per_control = 7 if model_name.lower() == 'phi4' else 3  # Phi4:7s, Llama3.1:3s
        
        num_controls = count_controls(excel_path)
        llm_time = num_controls * llm_time_per_control  # Total LLM processing time
        
        # Total ETA includes pre_llm_time, llm_time, and qualifier_time
        total_eta = pre_llm_time + llm_time + qualifier_time
        progress_data[task_id]['eta'] = int(total_eta)
        progress_data[task_id]['progress'] = 0.0
        progress_data[task_id]['status'] = "Initializing..."
        progress_data[task_id]['download_url'] = None
        progress_data[task_id]['error'] = None
        progress_data[task_id]['num_controls'] = num_controls
        progress_data[task_id]['cancelled'] = False
        
        # Progress increments
        progress_increment_pre_llm = 20.0 / pre_llm_steps  # 20% divided by 6 steps ≈ 3.33% per step
        progress_increment_llm = 70.0 / num_controls if num_controls > 0 else 0  # 70% allocated for LLM
        progress_increment_qualifier = 10.0  # 10% allocated for qualifiers
        
        # Thread-safe lock for progress_data
        from threading import Lock
        progress_lock = Lock()
        
        # Helper function to check for cancellation
        def check_cancel(tid):
            if progress_data[tid].get('cancelled'):
                logging.info(f"Task {tid} has been cancelled.")
                with progress_lock:
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
        
        # Ensure Control IDs are provided
        if not control_id.strip():
            raise ValueError("No Control IDs were selected. Please provide Control IDs before continuing.")
        
        for idx, step_description in enumerate(pre_llm_phase_descriptions):
            check_cancel(task_id)
            with progress_lock:
                progress_data[task_id]['status'] = step_description
            logging.info(f"Task {task_id}: {step_description}, ETA: {format_eta(progress_data[task_id]['eta'])}")
            
            if idx == 0:
                # 1: Extracting full text for qualifiers
                full_text_output_path = os.path.join(app.config['RAG_OUTPUTS'], f"full_text_{task_id}.txt")
                full_extracted_text = extract_text_from_pdf(pdf_path, qualifiers_start_page, qualifiers_end_page, full_text_output_path)
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(full_text_output_path):
                    raise FileNotFoundError(f"Extracted full text file not found at {full_text_output_path}")
                with progress_lock:
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
                with progress_lock:
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
                with progress_lock:
                    progress_data[task_id]['progress'] += progress_increment_pre_llm
            
            elif idx == 3:
                # 4: Extracting controls' text
                controls_output_text_path = os.path.join(app.config['RAG_OUTPUTS'], f"controls_text_{task_id}.txt")
                controls_extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page, controls_output_text_path)
                sleep_seconds(task_id, pre_llm_step_time)
                if not os.path.exists(controls_output_text_path):
                    raise FileNotFoundError(f"Controls extracted text file not found at {controls_output_text_path}")
                with progress_lock:
                    progress_data[task_id]['progress'] += progress_increment_pre_llm
            
            elif idx == 4:
                # 5: Chunking controls' text
                control_ids_raw = control_id
                control_ids = [cid.strip() for cid in control_ids_raw.split(',') if cid.strip()]
                
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
                with progress_lock:
                    progress_data[task_id]['progress'] += progress_increment_pre_llm
            
            elif idx == 5:
                # 6: Building RAG system for controls
                faiss_index_controls_path = os.path.join(app.config['RAG_OUTPUTS'], f"faiss_index_controls_{task_id}.idx")
                # Reuse 'control_patterns' from chunking
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
                with progress_lock:
                    progress_data[task_id]['progress'] += progress_increment_pre_llm
        
        # ---------------------- LLM Analysis ----------------------
        # Determine Control IDs order
        control_ids_order = [cid.strip() for cid in control_id.split(',') if cid.strip()]
        logging.info(f"Control IDs order for page range determination: {control_ids_order}")
        
        # Detect Control IDs and their pages
        # To handle shared regex patterns, group Control IDs by their regex
        repeating_patterns = identify_control_ids(pdf_path)  # List of dicts with 'Regex Pattern' and 'Example Control ID'
        regex_to_cids = {}
        for pattern_dict in repeating_patterns:
            regex = pattern_dict.get("Regex Pattern")
            cid = pattern_dict.get("Example Control ID")
            if regex and cid and cid in control_ids_order:
                regex_to_cids.setdefault(regex, []).append(cid)
        
        if not regex_to_cids:
            raise ValueError("No matching Control IDs found in the document based on the selected Control IDs.")
        
        control_id_pages = detect_control_id_pages(pdf_path, regex_to_cids)
        
        # Determine page range based on Control IDs with prioritization
        start_page, end_page = determine_page_range(control_id_pages, regex_to_cids)
        logging.info(f"Determined page range based on Control IDs: Start Page = {start_page}, End Page = {end_page}")
        
        # Process control framework with RAG
        check_cancel(task_id)
        with progress_lock:
            progress_data[task_id]['status'] = "Processing control framework with RAG..."
        logging.info(f"Task {task_id}: Processing control framework with RAG. ETA: {format_eta(progress_data[task_id]['eta'])}")
        processed_framework_path = os.path.join(app.config['RAG_OUTPUTS'], f"cybersecurity_framework_with_answers_{task_id}.xlsx")
        faiss_index_controls_path = os.path.join(app.config['RAG_OUTPUTS'], f"faiss_index_controls_{task_id}.idx")
        df_control_chunks_path = os.path.join(app.config['RAG_OUTPUTS'], f"df_control_chunks_{task_id}.csv")
        
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
        
        # ---------------------- LLM Processing ----------------------
        # Analyze controls with LLM
        with progress_lock:
            progress_data[task_id]['status'] = "Analyzing controls with LLM..."
        logging.info(f"Task {task_id}: Analyzing controls with LLM. ETA: {format_eta(progress_data[task_id]['eta'])}")
        analysis_df = load_responses(processed_framework_path)
        analyzed_rows = []
        
        for i, row in analysis_df.iterrows():
            check_cancel(task_id)
            with progress_lock:
                progress_data[task_id]['status'] = f"Analyzing control {i+1} of {num_controls} with LLM..."
            logging.info(f"Task {task_id}: Analyzing control {i+1}/{num_controls}. ETA: {format_eta(progress_data[task_id]['eta'])}")
            
            # Sleep for LLM processing time per control
            sleep_seconds(task_id, llm_time_per_control)
            # **Removed redundant ETA decrement**
            # progress_data[task_id]['eta'] -= llm_time_per_control  # Removed to prevent double decrement
            
            # Process exactly one control at a time, passing the chosen model
            processed_row = process_controls(pd.DataFrame([row]), model_name=model_name)
            analyzed_rows.append(processed_row)
            with progress_lock:
                progress_data[task_id]['progress'] += progress_increment_llm
        
        analyzed_df = pd.concat(analyzed_rows, ignore_index=True) if analyzed_rows else pd.DataFrame()
        
        # **Call remove_not_met_controls after LLM analysis and Explanation is populated**
        analyzed_df = remove_not_met_controls(analyzed_df)
        
        # Save analyzed_df to Excel
        analysis_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"Framework_Analysis_Completed_{task_id}.xlsx")
        analyzed_df.to_excel(analysis_output_path, index=False)
        logging.info(f"Task {task_id}: LLM analysis completed at {analysis_output_path}")
        check_cancel(task_id)
        
        # ---------------------- Merge and Finalize ----------------------
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
        with progress_lock:
            progress_data[task_id]['progress'] = 90.0
        rename_sheet_to_soc_mapping(final_output_path)
        check_cancel(task_id)
        
        # ---------------------- Qualifier Checks ----------------------
        # Qualifier checks
        with progress_lock:
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
        
        # ---------------------- Qualifier Time Processing ----------------------
        # Sleep for qualifier_time to account for processing time
        with progress_lock:
            progress_data[task_id]['status'] = "Finalizing qualifier processing..."
        logging.info(f"Task {task_id}: Finalizing qualifier processing. ETA: {format_eta(progress_data[task_id]['eta'])}")
        sleep_seconds(task_id, qualifier_time)
        with progress_lock:
            progress_data[task_id]['eta'] -= qualifier_time
            progress_data[task_id]['progress'] += progress_increment_qualifier
        check_cancel(task_id)
        
        # ---------------------- Executive Summary ----------------------
        # Create Executive Summary in a new final file
        with progress_lock:
            progress_data[task_id]['status'] = "Creating Executive Summary..."
        logging.info(f"Task {task_id}: Creating Executive Summary. ETA: {format_eta(progress_data[task_id]['eta'])}")
        
        # Generate the final filename based on input filenames
        soc_report_basename = os.path.splitext(os.path.basename(soc_report_filename))[0]
        framework_basename = os.path.splitext(os.path.basename(framework_filename))[0]
        final_filename = f"{soc_report_basename}_Baselined_Vs_{framework_basename}.xlsx"
        summary_output_path = os.path.join(app.config['EXCEL_FOLDER'], final_filename)
        
        create_executive_summary(final_output_path, summary_output_path)
        logging.info(f"Task {task_id}: Executive Summary created at {summary_output_path}")
        
        # Set the download URL to the final summarized Excel file
        with progress_lock:
            progress_data[task_id]['download_url'] = f"https://g6lxt0v21br58e-5000.proxy.runpod.net/download/{final_filename}"
        
        with progress_lock:
            progress_data[task_id]['progress'] = 100.0
            progress_data[task_id]['eta'] = 0
            progress_data[task_id]['status'] = "Task completed successfully."
        logging.info(f"Task {task_id}: Task completed successfully.")
    
    except Exception as e:
        logging.error(f"Error in background_process (task_id: {task_id}): {e}", exc_info=True)
        if "cancelled" in str(e).lower():
            # Mark the task as cancelled
            with progress_lock:
                progress_data[task_id]['cancelled'] = True
                progress_data[task_id]['status'] = "Cancelled"
                progress_data[task_id]['progress'] = 0
                progress_data[task_id]['eta'] = 0
                progress_data[task_id]['error'] = "Task was cancelled by the user."
        else:
            with progress_lock:
                progress_data[task_id]['status'] = "Error occurred."
                progress_data[task_id]['error'] = str(e)
                progress_data[task_id]['eta'] = 0

# -------------------------------------------------------
# NEW ENDPOINT: /detect_control_ids
# -------------------------------------------------------
@app.route('/detect_control_ids', methods=['POST'])
def detect_control_ids_endpoint():
    """
    1) Accepts a PDF file.
    2) Calls identifier.process_pdf(...) to detect repeating Control IDs.
    3) Detects the pages where these Control IDs appear.
    4) Returns Control IDs and their page numbers in JSON for the frontend.
    """
    logging.info("Received request to /detect_control_ids endpoint.")
    try:
        pdf_file = request.files.get('pdf_file')
        if not pdf_file:
            raise ValueError("Missing required file: pdf_file")

        # Secure filename and save
        filename_pdf = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_pdf)
        pdf_file.save(pdf_path)
        logging.info(f"PDF file saved to {pdf_path}")

        # Identify control IDs by calling identifier.process_pdf
        repeating_patterns = identify_control_ids(pdf_path)  # expects to return list of dicts

        # Group Control IDs by their regex patterns
        regex_to_cids = {}
        for pattern_dict in repeating_patterns:
            regex = pattern_dict.get("Regex Pattern")
            cid = pattern_dict.get("Example Control ID")
            if regex and cid:
                regex_to_cids.setdefault(regex, []).append(cid)

        # Extract Control IDs from repeating patterns
        control_ids = [cid for cids in regex_to_cids.values() for cid in cids]

        # Detect pages where Control IDs appear
        control_id_pages = detect_control_id_pages(pdf_path, regex_to_cids)

        # Example return structure:
        # {
        #   "repeating_patterns": [...],
        #   "control_id_pages": {
        #       "AC-32": [56],
        #       "CC1.1": [47, 72],
        #       "AC-13a": [57, 74],
        #       ...
        #   }
        # }
        return jsonify({
            "repeating_patterns": repeating_patterns,
            "control_id_pages": control_id_pages
        }), 200

    except Exception as e:
        logging.error(f"Error in /detect_control_ids endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# -------------------------------------------------------
# Existing Endpoint: /process_all
# -------------------------------------------------------
@app.route('/process_all', methods=['POST'])
def process_all():
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