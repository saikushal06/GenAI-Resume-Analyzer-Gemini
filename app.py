import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer & Job Matcher")
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    background-color: #4F46E5;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 220px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #3730A3;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}

h1, h2, h3 {
    color: #111827;
}

</style>
""", unsafe_allow_html=True)
st.write("Upload your resume and paste a job description to check ATS match, missing skills, and improvement suggestions.")


def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


skills_list = [
    "python", "java", "c++", "sql", "mysql", "html", "css", "javascript",
    "react", "node.js", "machine learning", "deep learning", "data analysis",
    "data science", "pandas", "numpy", "scikit-learn", "tensorflow",
    "keras", "nlp", "power bi", "tableau", "excel", "statistics",
    "communication", "problem solving", "leadership", "teamwork",
    "git", "github", "streamlit", "flask", "django", "firebase",
    "cloud", "aws", "azure", "docker"
]


def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return sorted(set(found_skills))


def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=250)

if st.button("Analyze Resume"):
    if resume_file is None:
        st.error("Please upload your resume PDF.")
    elif not job_description.strip():
        st.error("Please paste the job description.")
    else:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(resume_file)

            cleaned_resume = clean_text(resume_text)
            cleaned_jd = clean_text(job_description)

            match_score = calculate_similarity(cleaned_resume, cleaned_jd)

            resume_skills = extract_skills(cleaned_resume)
            jd_skills = extract_skills(cleaned_jd)

            missing_skills = sorted(set(jd_skills) - set(resume_skills))

        st.success("Analysis Completed!")

        col1, col2, col3 = st.columns(3)

        col1.metric("ATS Match Score", f"{match_score}%")
        st.progress(match_score / 100)
        col2.metric("Resume Skills Found", len(resume_skills))
        col3.metric("Missing Skills", len(missing_skills))

        st.subheader("✅ Skills Found in Resume")
        if resume_skills:
            st.write(", ".join(resume_skills))
        else:
            st.warning("No major skills found from the predefined skill list.")

        st.subheader("🎯 Skills Required in Job Description")
        if jd_skills:
            st.write(", ".join(jd_skills))
        else:
            st.warning("No major skills found in the job description.")

        st.subheader("⚠️ Missing Skills")
        if missing_skills:
            for skill in missing_skills:
                st.write(f"- {skill}")
        else:
            st.success("Great! Your resume covers the main skills from the job description.")

        st.subheader("💡 Resume Improvement Suggestions")

        if match_score >= 75:
            st.success("Your resume is strongly aligned with this job description.")
        elif match_score >= 50:
            st.info("Your resume has a decent match. Add more job-specific keywords and measurable achievements.")
        else:
            st.warning("Your resume needs improvement. Add relevant skills, projects, tools, and keywords from the job description.")

        if missing_skills:
            st.write("Add or improve these missing skills if you genuinely know them:")
            st.write(", ".join(missing_skills))

        st.write("Use action verbs like built, developed, implemented, analyzed, optimized, and deployed.")
        st.write("Add project impact, technologies used, and measurable results wherever possible.")