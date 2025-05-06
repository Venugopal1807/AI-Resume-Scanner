
from typing import Dict, List
import streamlit as st
import base64
from utils.document_processor import DocumentProcessor
from utils.nlp_analyzer import NLPAnalyzer
from utils.ml_scorer import MLScorer

def load_css() -> None:
    """Load custom CSS styles."""
    with open('styles/main.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_sample_job_description() -> str:
    """Return a sample job description."""
    return """
    Senior Software Engineer
    Requirements:
    - 5+ years of experience in Python development
    - Strong knowledge of web frameworks (Django, Flask)
    - Experience with cloud platforms (AWS, GCP)
    - Background in machine learning and data analysis
    - Excellent communication and team collaboration skills
    Bachelor's degree in Computer Science or related field required
    """

def display_instructions() -> None:
    """Display usage instructions."""
    st.markdown('<div class="instructions-section">', unsafe_allow_html=True)
    st.markdown("## How to Use This Tool")
    st.markdown("""
    Follow these simple steps to analyze resumes:

    1. **Job Description**
       - Click '📝 Load Sample Job Description' for a quick start, or
       - Enter your own job description in the text area

    2. **Upload Resumes**
       - Click 'Browse files' to upload resumes
       - Supported formats: PDF, DOC, DOCX
       - You can upload multiple resumes at once

    3. **Process and Analyze**
       - Click 'Process Resumes' to start the analysis
       - Wait for the progress bar to complete
       - Review the detailed results for each resume

    4. **Review Results**
       - Overall match score
       - Skills analysis (matched and missing skills)
       - Education and experience evaluation
       - Document statistics
       - Named entities (organizations, people, locations)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def display_resume_results(resume_data: Dict, idx: int) -> None:
    """Display analysis results for a single resume."""
    with st.expander(f"#{idx} - {resume_data['filename']} (Score: {resume_data['scores']['overall_score']}%)"):
        # Score Breakdown
        st.subheader("Score Breakdown")
        cols = st.columns(4)
        metrics = [
            ("Content Match", resume_data['scores']['content_similarity']),
            ("Skills Match", resume_data['scores']['skills_match']),
            ("Education", resume_data['scores']['education_level']),
            ("Experience", resume_data['scores']['experience_level'])
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, f"{value}%")

        # Skills Analysis
        st.subheader("Skills Analysis")
        col1, col2 = st.columns(2)
        with col1:
            display_skills("Matched Skills", resume_data['scores']['matched_skills'])
        with col2:
            display_skills("Missing Skills", resume_data['scores']['missing_skills'])

        # Document Statistics
        display_document_stats(resume_data['stats'])
        
        # Named Entities
        display_entities(resume_data['entities'])

def display_skills(title: str, skills: List[str]) -> None:
    """Display skills with formatting."""
    st.markdown(f"##### {title}")
    if skills:
        skill_class = "matched-skill" if "Matched" in title else "missing-skill"
        skills_html = ''.join([f'<span class="skills-tag {skill_class}">{skill}</span>' for skill in skills])
        st.markdown(f'<div class="skills-container">{skills_html}</div>', unsafe_allow_html=True)
    else:
        st.write("None found")

def display_document_stats(stats: Dict) -> None:
    """Display document statistics."""
    st.subheader("Document Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", stats['word_count'])
    with col2:
        st.metric("Sentence Count", stats['sentence_count'])
    with col3:
        st.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")

def display_entities(entities: Dict) -> None:
    """Display named entities."""
    st.subheader("Named Entities")
    col1, col2, col3 = st.columns(3)
    entity_types = [("Organizations", "ORGANIZATION"), ("People", "PERSON"), ("Locations", "GPE")]
    
    for col, (title, key) in zip([col1, col2, col3], entity_types):
        with col:
            st.markdown(f"##### {title}")
            if entities[key]:
                for entity in entities[key]:
                    st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
            else:
                st.write("None found")

def main():
    """Main application function."""
    load_css()
    st.markdown('<h1 class="main-title">AI Resume Screening System</h1>', unsafe_allow_html=True)

    # Job Description Input
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("Job Description")

    if st.button("📝 Load Sample Job Description", help="Click to load a sample job description"):
        st.session_state['job_description'] = load_sample_job_description()
        st.rerun()

    job_description = st.text_area(
        "Enter the job description",
        height=200,
        value=st.session_state.get('job_description', ''),
        placeholder="Paste the job description here..."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize components
    doc_processor = DocumentProcessor()
    nlp_analyzer = NLPAnalyzer()
    ml_scorer = MLScorer()

    # Resume Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("Resume Upload")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOC, or DOCX format)",
        type=['pdf', 'doc', 'docx'],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files

    # Process Resumes
    if uploaded_files and job_description and st.button("Process Resumes"):
        with st.spinner('Processing resumes...'):
            try:
                results = process_resumes(uploaded_files, job_description, doc_processor, nlp_analyzer, ml_scorer)
                display_results(results)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    display_instructions()

def process_resumes(uploaded_files, job_description, doc_processor, nlp_analyzer, ml_scorer) -> List[Dict]:
    """Process uploaded resumes and return results."""
    results = []
    progress_bar = st.progress(0)

    for idx, file in enumerate(uploaded_files):
        # Extract and analyze resume
        resume_text = doc_processor.extract_text(file)
        doc_stats = doc_processor.get_document_stats(resume_text)
        entities = nlp_analyzer.extract_entities(resume_text)
        scores = ml_scorer.calculate_advanced_scores(job_description, resume_text)

        results.append({
            'filename': file.name,
            'text': resume_text,
            'stats': doc_stats,
            'entities': entities,
            'scores': scores
        })

        progress_bar.progress(int((idx + 1) / len(uploaded_files) * 100))

    return sorted(results, key=lambda x: x['scores']['overall_score'], reverse=True)

def display_results(results: List[Dict]) -> None:
    """Display all resume analysis results."""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.header("Analysis Results")
    for idx, result in enumerate(results, 1):
        display_resume_results(result, idx)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()