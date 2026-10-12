# Flash Grader

An automated, reproducible exam grading and rubric management system for handwritten, code, and text student answer scripts.

---

## Features

- **Objective Evaluation**: Standardized, reproducible grading criteria across runs.
- **Rubric Generation**: Automatically solves question papers and generates reference marking schemes, or ingests existing answer keys.
- **Multimodal Ingestion**: Supports exam papers and student submissions in PDF or image formats (`.jpg`, `.png`, `.heic`).
- **Detailed Feedback**: Per-question scoring breakdown with constructive reasoning and criteria explanations.
- **Multi-Format Export**: Export evaluation results in Markdown (`.md`), CSV (`.csv`), or structured JSON (`.json`).

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- An API Key with access to Google GenAI services.

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Set API Key (Optional)
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your-api-key"

# Linux / macOS
export GOOGLE_API_KEY="your-api-key"
```
*(You can also provide the key directly within the app sidebar).*

### 4. Run Application
```bash
streamlit run app.py
```

---

## Docker Deployment

To build and run with Docker:

```bash
docker build -t flash-grader .
docker run -p 8080:8080 -e GOOGLE_API_KEY="your-api-key" flash-grader
```

Access the application at `http://localhost:8080`.
