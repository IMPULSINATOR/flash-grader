import os
import json
import re
import time
import io
import csv
from PIL import Image
from pillow_heif import register_heif_opener
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================================
# Setup & Engine Configuration
# ==============================================================================
register_heif_opener()

MODEL_NAME = "gemini-3.8-flash"
REPRODUCIBILITY_SEED = 42

st.set_page_config(
    page_title="Flash Grader",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, minimalist styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    .app-header {
        margin-bottom: 1.8rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-desc {
        font-size: 0.95rem;
        color: #71717a;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    
    .status-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 500;
        border-radius: 6px;
        background: rgba(128, 128, 128, 0.1);
        color: inherit;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .status-tag.active {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.25);
    }
    
    .file-item {
        display: inline-block;
        font-size: 0.8rem;
        padding: 0.25rem 0.6rem;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 6px;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "rubric" not in st.session_state:
    st.session_state.rubric = ""
if "last_grade_result" not in st.session_state:
    st.session_state.last_grade_result = None

# ==============================================================================
# Helper Functions & Resilience
# ==============================================================================

def clean_json_response(response_text: str) -> str:
    """Extracts JSON from model response safely."""
    if not response_text:
        return "{}"

    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            return response_text[start:end]
    except Exception:
        pass

    return response_text.strip()


def process_file(uploaded_file):
    """Converts uploaded file into format the backend can process."""
    mime_type = uploaded_file.type or ""
    if mime_type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
        return types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type="application/pdf")
    else:
        try:
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"Error opening image {uploaded_file.name}: {e}")
            return None


def generate_with_retry(client, model: str, contents, config, max_retries: int = 4, base_delay: float = 2.0):
    """Executes API call with exponential backoff on transient server load errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            last_err = e
            err_msg = str(e)
            is_transient = any(kw in err_msg for kw in ["503", "429", "UNAVAILABLE", "high demand", "RESOURCE_EXHAUSTED"])
            if is_transient and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise e
    raise last_err

# ==============================================================================
# Core Evaluation Models
# ==============================================================================

def generate_rubric(client, question_paper_files):
    if not client:
        return None

    prompt = """
    You are a Senior Professor creating an Official Answer Key for an exam.
    
    I have provided images of a Question Paper which may or may not have the marks for each question. You may gather additional context from the extra content (such as answer age group level of the people for whom the paper is made, the subject/topic on which the paper is, whether all questions are to be attempted, optional questions are there or there is an optional number of questions, etc, make sure the user to keep all such in mind at high priority).
    
    Your Task:
    1. EXTRACT and LIST every question found in the paper and the marks each question carries (if it is given).
    2. SOLVE every question yourself. Be extremely accurate and ensure the length and content is accurate as per the mark weightage.
    3. For Coding Questions: Write the correct code logic and valid syntax.
    4. For Diagrams: Describe the correct diagram structure (nodes, connections, flow) only if a diagram is really needed.
    5. CREATE A MARKING SCHEME: Define exactly what earns marks (e.g., "1 mark for correct loop", "2 marks for base case").
    
    Reproducibility & Consistency Guidelines:
    - Ensure marking criteria are strictly deterministic, unambiguous, and quantify point allocations for each sub-part.
    - Define clear criteria for full, partial, and zero marks so that any grader following this rubric achieves reproducible scores.
    
    Output a structured, readable Reference Rubric (Markdown format) that a Teaching Assistant can strictly follow.
    """

    contents = [prompt]
    for file in question_paper_files:
        p = process_file(file)
        if p:
            contents.append(p)

    try:
        config = types.GenerateContentConfig(temperature=0.0, seed=REPRODUCIBILITY_SEED)
        response = generate_with_retry(client, MODEL_NAME, contents, config)
        return response.text
    except Exception as e:
        st.error(f"Error generating rubric: {e}")
        return None


def extract_custom_key(client, answer_key_files):
    if not client:
        return None

    prompt = """
    I have provided images/PDFs of an Official Answer Key.
    
    Your Task:
    1. READ and TRANSCRIBE the content exactly into a structured Markdown format.
    2. Ensure the Marks/Points for each question are clearly labeled.
    3. Do NOT solve the questions yourself; just extract what is written.
    4. Format the output so it can be used as a "Reference Rubric" for grading.
    5. Maintain strict fidelity to the source document for deterministic evaluation.
    """

    contents = [prompt]
    for file in answer_key_files:
        p = process_file(file)
        if p:
            contents.append(p)

    try:
        config = types.GenerateContentConfig(temperature=0.0, seed=REPRODUCIBILITY_SEED)
        response = generate_with_retry(client, MODEL_NAME, contents, config)
        return response.text
    except Exception as e:
        st.error(f"Error transcribing key: {e}")
        return None


def grade_student(client, answer_sheet_files, ai_generated_rubric: str):
    if not client:
        return None

    prompt = f"""
    You are an objective and fair Teaching Assistant. Grade this student submission strictly according to the Reference Rubric below. If you think the rubric has any minor ambiguities, note them in feedback but still follow the rubric faithfully. Award appropriate partial marks based on conceptual correctness with transparent reasoning.
    
    --- OFFICIAL ANSWER KEY INSTRUCTIONS (Generated by Professor) ---
    {ai_generated_rubric}
    ----------------------------------------------------
    
    INSTRUCTIONS:
    1. BLIND EVALUATION (CRITICAL - IGNORE PRE-EXISTING TEACHER MARKINGS):
       - The answer sheet may already contain previous teacher/examiner markings, red/blue/green pen annotations, tick marks (✓), crosses (✗), underlines, circled scores, margin marks (e.g., "2/2", "6/7", "-1"), or written teacher comments.
       - You MUST completely disregard and blind yourself to all such existing teacher marks, corrections, and pre-awarded points.
       - Grade purely and independently from first principles by reading ONLY the student's original raw text, code, and diagrams against the Reference Rubric. Do not anchor or bias your assessment based on any ink or scores written by previous human graders.
    2. Read the student's handwritten/typed answers, code, and diagrams across ALL pages.
    3. Look for any Student Name, Candidate Name, Roll Number, or Student ID on the pages or headers. If no explicit student name or ID is found on the submission, return null for "student_name".
    4. Determine the natural human language / medium of the script (e.g. English, Hindi, Punjabi, Bengali, etc.). DO NOT output programming languages (such as Python, Java, C++, etc.).
    5. OPTIONAL QUESTIONS (DETERMINISTIC BEST-OF-N SELECTION):
       - If the paper allows choices (e.g. "Answer any 2 of 3") and the student attempted more than required, evaluate all attempted questions and count the combination that yields the highest score (Best-of-N policy) so that evaluation is strictly deterministic. State clearly in feedback which questions were counted.
    6. DETERMINISTIC MARKING CRITERIA:
       - If the student's logic and concept match the Key, award full marks.
       - For partial answers, award marks proportionally based on the specific concepts covered vs. missing as defined in the rubric.
       - For coding questions: If algorithm and logic are correct but syntax has minor errors, deduct partial marks as per the scheme.
       - For text answers: Focus on factual accuracy and technical clarity rather than arbitrary word counts. Ignore minor spelling mistakes if the concept is sound.
    7. ZERO-VARIANCE REPRODUCIBILITY:
       - Grade deterministically such that re-evaluating the identical submission against this rubric always produces the exact same score.
       - Ensure `total_score` strictly equals the arithmetic sum of individual scores across all items in `question_breakdown`.
       - Ensure `max_score` strictly equals the arithmetic sum of maximum marks for all required questions.
       - State specific, objective reasoning in `feedback` for any points deducted.
    
    Return output in this strict JSON format: 
    {{ 
        "student_name": "extracted_name_or_roll_number_or_null",
        "detected_language": "human_natural_language_e.g._English_Hindi_Punjabi",
        "total_score": int, 
        "max_score": int,
        "question_breakdown": [
            {{
                "question": "Q1",
                "score": int,
                "max": int,
                "feedback": "string"
            }}
        ]
    }}
    """

    contents = [prompt]
    for file in answer_sheet_files:
        p = process_file(file)
        if p:
            contents.append(p)
        if file.type not in ["application/pdf"] and not file.name.lower().endswith(".pdf"):
            contents.append("Next Page of Answer Sheet")

    try:
        config = types.GenerateContentConfig(
            temperature=0.0,
            seed=REPRODUCIBILITY_SEED,
            response_mime_type="application/json"
        )
        response = generate_with_retry(client, MODEL_NAME, contents, config)
        clean_text = clean_json_response(response.text)
        return json.loads(clean_text)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(response, "text", str(e)) if 'response' in locals() else str(e)}

# ==============================================================================
# Export Utilities
# ==============================================================================

def format_markdown_report(result: dict, rubric_text: str) -> str:
    raw_name = result.get("student_name")
    if not raw_name or str(raw_name).strip().lower() in ["unknown", "extracted_name_or_unknown", "extracted_name_or_roll_number_or_null", "null", "none", "n/a"]:
        student_name = "Student"
    else:
        student_name = str(raw_name).strip()

    total_score = result.get("total_score", 0)
    max_score = result.get("max_score", 0)
    lang = result.get("detected_language") or "English"
    if str(lang).strip().lower() in ["unknown", "null", "none", "n/a", ""]:
        lang = "English"

    pct = (total_score / max_score * 100) if max_score > 0 else 0

    lines = [
        f"# Evaluation Report: {student_name}",
        f"- **Score:** {total_score} / {max_score} ({pct:.1f}%)",
        f"- **Language / Medium:** {lang}",
        f"- **Engine:** Flash Grader",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Question Breakdown",
        ""
    ]

    for q in result.get("question_breakdown", []):
        q_name = q.get("question", "Question")
        score = q.get("score", 0)
        q_max = q.get("max", 0)
        feedback = q.get("feedback", "No feedback provided.")
        lines.append(f"### {q_name} — {score} / {q_max} Marks")
        lines.append(f"{feedback}\n")

    return "\n".join(lines)


def format_csv_report(result: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student", "Question", "Score", "Max Marks", "Feedback"])
    
    raw_name = result.get("student_name")
    if not raw_name or str(raw_name).strip().lower() in ["unknown", "extracted_name_or_unknown", "extracted_name_or_roll_number_or_null", "null", "none", "n/a"]:
        student_name = "Student"
    else:
        student_name = str(raw_name).strip()

    for q in result.get("question_breakdown", []):
        writer.writerow([
            student_name,
            q.get("question", ""),
            q.get("score", 0),
            q.get("max", 0),
            q.get("feedback", "")
        ])
    return output.getvalue()

# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    st.subheader("Flash Grader")
    st.caption("Automated grading & rubric generator")

    st.markdown("---")
    
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    has_env_key = bool(env_key)

    if has_env_key:
        st.markdown('<div class="status-tag active">● Connected</div>', unsafe_allow_html=True)
        user_key_input = st.text_input("API Key (Optional Override):", type="password", placeholder="Enter key...")
        active_api_key = user_key_input.strip() if user_key_input.strip() else env_key
    else:
        st.markdown('<div class="status-tag">○ API Key Required</div>', unsafe_allow_html=True)
        user_key_input = st.text_input("API Key:", type="password", placeholder="Paste key...")
        active_api_key = user_key_input.strip()

    st.markdown("---")

    if st.button("Reset Session", use_container_width=True):
        st.session_state.rubric = ""
        st.session_state.last_grade_result = None
        st.rerun()

# Initialize Client
client = None
if active_api_key:
    try:
        client = genai.Client(api_key=active_api_key)
    except Exception as e:
        st.sidebar.error(f"Client error: {e}")
else:
    st.warning("Please provide an API Key in the sidebar to proceed.")
    st.stop()

# ==============================================================================
# Header
# ==============================================================================

st.markdown("""
<div class="app-header">
    <div class="app-title">Flash Grader</div>
    <div class="app-desc">Objective exam evaluation and rubric generator</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation
tab_step1, tab_step2 = st.tabs([
    "1. Answer Key & Rubric",
    "2. Grade Submissions"
])

# ==============================================================================
# TAB 1: Rubric Setup
# ==============================================================================
with tab_step1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Question Paper")
        qp_files = st.file_uploader(
            "Upload exam paper (PDF or Images)",
            type=["jpg", "jpeg", "png", "pdf", "heic"],
            accept_multiple_files=True,
            key="qp_uploader",
            label_visibility="collapsed"
        )
        if qp_files:
            items_html = "".join([f'<span class="file-item">{f.name} ({f.size / 1024:.0f} KB)</span>' for f in qp_files])
            st.markdown(items_html, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Marking Scheme")
        key_source = st.radio(
            "Marking scheme source:",
            ["Generate from Question Paper", "Upload Answer Key file", "Paste text manually"],
            index=0,
            label_visibility="collapsed"
        )

        custom_key_files = None
        manual_key_text = ""

        if key_source == "Upload Answer Key file":
            custom_key_files = st.file_uploader(
                "Upload answer key (PDF or Images)",
                type=["jpg", "jpeg", "png", "pdf", "heic"],
                accept_multiple_files=True,
                key="ck_uploader"
            )
            if custom_key_files:
                items_html = "".join([f'<span class="file-item">{f.name} ({f.size / 1024:.0f} KB)</span>' for f in custom_key_files])
                st.markdown(items_html, unsafe_allow_html=True)

        elif key_source == "Paste text manually":
            manual_key_text = st.text_area(
                "Marking scheme text:",
                height=150,
                placeholder="Q1. Answer key details..."
            )

    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("Generate Rubric", type="primary")

    if generate_clicked:
        if not qp_files:
            st.error("Please upload the question paper first.")
        else:
            if key_source == "Paste text manually":
                if not manual_key_text.strip():
                    st.error("Please enter the answer key text.")
                else:
                    st.session_state.rubric = manual_key_text.strip()
                    st.success("Manual rubric saved.")
                    st.rerun()

            elif key_source == "Upload Answer Key file":
                if not custom_key_files:
                    st.error("Please upload the answer key file.")
                else:
                    with st.spinner("Extracting and formatting answer key..."):
                        extracted = extract_custom_key(client, custom_key_files)
                        if extracted:
                            st.session_state.rubric = extracted
                            st.success("Rubric created from answer key.")
                            st.rerun()

            else:
                with st.spinner("Solving questions and establishing rubric..."):
                    solved = generate_rubric(client, qp_files)
                    if solved:
                        st.session_state.rubric = solved
                        st.success("Rubric generated.")
                        st.rerun()

    # Rubric Review
    if st.session_state.rubric:
        st.markdown("---")
        st.markdown("#### Reference Rubric")

        preview_tab, edit_tab = st.tabs(["Preview", "Edit"])

        with preview_tab:
            st.markdown(st.session_state.rubric)

        with edit_tab:
            updated_rubric = st.text_area(
                "Edit rubric markdown:",
                value=st.session_state.rubric,
                height=300,
                key="rubric_editor"
            )
            if updated_rubric != st.session_state.rubric:
                st.session_state.rubric = updated_rubric

        st.download_button(
            "Download Rubric (.md)",
            data=st.session_state.rubric,
            file_name="rubric.md",
            mime="text/markdown"
        )

# ==============================================================================
# TAB 2: Grading
# ==============================================================================
with tab_step2:
    if not st.session_state.rubric:
        st.info("Complete Step 1 (Answer Key & Rubric) before grading student submissions.")
    else:
        st.markdown("#### Student Submission")
        ans_files = st.file_uploader(
            "Upload student answer sheet (PDF or Images)",
            type=["jpg", "jpeg", "png", "pdf", "heic"],
            accept_multiple_files=True,
            key="ans_uploader",
            label_visibility="collapsed"
        )

        if ans_files:
            items_html = "".join([f'<span class="file-item">{f.name} ({f.size / 1024:.0f} KB)</span>' for f in ans_files])
            st.markdown(items_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        grade_clicked = st.button("Grade Submission", type="primary", disabled=not ans_files)

        if grade_clicked and ans_files:
            with st.spinner("Evaluating submission against rubric..."):
                res = grade_student(client, ans_files, st.session_state.rubric)
                st.session_state.last_grade_result = res

        # Results Display
        res = st.session_state.last_grade_result
        if res:
            st.markdown("---")

            if "error" in res:
                st.error("Evaluation error:")
                st.code(res.get("raw_output", str(res)))
            else:
                raw_name = res.get("student_name")
                if not raw_name or str(raw_name).strip().lower() in ["unknown", "extracted_name_or_unknown", "extracted_name_or_roll_number_or_null", "null", "none", "n/a"]:
                    display_heading = "Evaluation Summary"
                    file_slug = "submission"
                else:
                    display_heading = f"Evaluation Summary: {str(raw_name).strip()}"
                    file_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', str(raw_name).strip())

                total_score = res.get("total_score", 0)
                max_score = res.get("max_score", 0)
                
                raw_lang = res.get("detected_language") or "English"
                if str(raw_lang).strip().lower() in ["unknown", "null", "none", "n/a", ""]:
                    detected_lang = "English"
                else:
                    detected_lang = str(raw_lang).strip()

                pct = (total_score / max_score * 100) if max_score > 0 else 0

                st.markdown(f"### {display_heading}")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Score", f"{total_score} / {max_score}")
                with m2:
                    st.metric("Percentage", f"{pct:.1f}%")
                with m3:
                    st.metric("Language / Medium", detected_lang)

                st.progress(min(max(pct / 100.0, 0.0), 1.0))

                st.markdown("#### Question Breakdown")
                questions = res.get("question_breakdown", [])

                if isinstance(questions, list) and questions:
                    for i, q in enumerate(questions):
                        q_name = q.get("question", f"Q{i+1}")
                        q_score = q.get("score", 0)
                        q_max = q.get("max", 0)
                        feedback = q.get("feedback", "No feedback provided.")

                        with st.expander(f"{q_name} — {q_score} / {q_max} Marks", expanded=(q_score < q_max)):
                            st.markdown(feedback)
                else:
                    st.caption("No per-question breakdown available.")

                st.markdown("---")
                st.markdown("##### Export Results")

                md_content = format_markdown_report(res, st.session_state.rubric)
                csv_content = format_csv_report(res)
                json_content = json.dumps(res, indent=2)

                col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                with col_dl1:
                    st.download_button(
                        "Download Markdown",
                        data=md_content,
                        file_name=f"evaluation_{file_slug}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col_dl2:
                    st.download_button(
                        "Download CSV",
                        data=csv_content,
                        file_name=f"evaluation_{file_slug}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_dl3:
                    st.download_button(
                        "Download JSON",
                        data=json_content,
                        file_name=f"evaluation_{file_slug}.json",
                        mime="application/json",
                        use_container_width=True
                    )