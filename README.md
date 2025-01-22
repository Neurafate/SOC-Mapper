```markdown
# SOC-AI Flask API

SOC-AI is a Flask-based API that streamlines the analysis of SOC reports and cybersecurity frameworks. It utilizes advanced text extraction, RAG (Retrieval-Augmented Generation) systems, and LLM (Large Language Model) processing to provide detailed compliance analysis, qualifiers, and executive summaries.

---

## Features

- **PDF Text Extraction:** Extracts text from SOC reports for processing.
- **Text Chunking:** Splits text into manageable chunks using regex patterns or fixed sizes.
- **Compliance Analysis:** Generates compliance scores and analysis using LLMs.
- **RAG System Integration:** Implements RAG for efficient retrieval and analysis.
- **Excel Outputs:** Produces formatted Excel files, including compliance scores and executive summaries.
- **Real-Time Progress Tracking:** Streams task progress updates via SSE.
- **Task Management:** Background processing, cancellation, and secure file downloads.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed below)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/soc-ai.git
   cd soc-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

---

## API Endpoints

### **POST /process_all**
Starts processing a SOC report and cybersecurity framework.

- **Required Parameters**:
  - `pdf_file`: SOC report (PDF).
  - `excel_file`: Cybersecurity framework (Excel).
  - `start_page`: Start page for text extraction (default: 1).
  - `end_page`: End page for text extraction (default: 10).
  - `control_id`: Comma-separated list of control IDs.
  - `model_name`: LLM model name (default: `llama3.1`).

- **Response**:
  - `task_id`: Unique identifier for the task.

---

### **GET /progress/<task_id>**
Streams real-time progress updates for a task.

- **Response**:
  - `progress`: Percentage completion.
  - `status`: Current task status.
  - `eta`: Estimated time remaining.
  - `download_url`: Link to the final output file (if completed).

---

### **POST /cancel_task/<task_id>**
Cancels a running task.

- **Response**:
  - Confirmation message.

---

### **GET /download/<filename>**
Downloads the final Excel output file.

---

## Directory Structure

- **`/uploads`**: Stores uploaded files.
- **`/results`**: Stores intermediate results.
- **`/excel_outputs`**: Stores final Excel files.
- **`/rag_outputs`**: Stores outputs related to the RAG system.

---

## Logging

Logs are saved to `LLM_analysis.log`. They include task progress, errors, and performance metrics.

---

## Requirements

- Flask
- Flask-CORS
- pandas
- openpyxl
- Werkzeug
- Other dependencies listed in `requirements.txt`

---

## Usage Example

1. Upload a SOC report and framework:
   ```bash
   curl -X POST http://127.0.0.1:5000/process_all \
   -F "pdf_file=@soc_report.pdf" \
   -F "excel_file=@framework.xlsx" \
   -F "start_page=1" \
   -F "end_page=10" \
   -F "control_id=CONTROL001,CONTROL002" \
   -F "model_name=llama3.1"
   ```

2. Check task progress:
   ```bash
   curl http://127.0.0.1:5000/progress/<task_id>
   ```

3. Download the final output:
   ```bash
   curl -O http://127.0.0.1:5000/download/<filename>
   ```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```

This single file can be directly copied into your `README.md`.