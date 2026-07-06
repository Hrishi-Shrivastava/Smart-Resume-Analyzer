import datetime
import io
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import pymysql
import streamlit as st

from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #FF4B4B;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }
    
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    h1 {
        font-weight: 800;
        letter-spacing: -1px;
    }
    </style>
""", unsafe_allow_html=True)

if "saved_resumes" not in st.session_state:
    st.session_state.saved_resumes = set()

DB_HOST = st.secrets.get("mysql", {}).get("host", "localhost")
DB_USER = st.secrets.get("mysql", {}).get("user", "root")
DB_PASS = st.secrets.get("mysql", {}).get("password", "")
DB_PORT = int(st.secrets.get("mysql", {}).get("port", 3306))
DB_NAME = st.secrets.get("mysql", {}).get("database", "sra_db")

def get_db_connection():
    try:
        ssl_config = {"verify_mode": None} if DB_PORT == 4000 else None
        return pymysql.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASS, 
            port=DB_PORT, 
            database=DB_NAME, 
            ssl=ssl_config,
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        return None

def initialize_database():
    try:
        ssl_config = {"verify_mode": None} if DB_PORT == 4000 else None
        base_conn = pymysql.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASS, 
            port=DB_PORT,
            ssl=ssl_config
        )
        with base_conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
        base_conn.commit()
        base_conn.close()

        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_data (
                    id INT AUTO_INCREMENT PRIMARY KEY, 
                    timestamp VARCHAR(50), 
                    name VARCHAR(200),
                    email VARCHAR(200), 
                    resume_score INT, 
                    predicted_domain VARCHAR(200),
                    skills LONGTEXT, 
                    match_score INT
                );
                """)
            conn.commit()
            conn.close()
    except Exception as e:
        pass

initialize_database()

def extract_contact_info(text_corpus, default_name="Candidate Profile"):
    name = default_name
    first_line = text_corpus.strip().split("\n")[0]
    if len(first_line.strip()) > 2 and len(first_line.split()) <= 4:
        name = first_line.strip()
        
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text_corpus)
    email = email_match.group(0) if email_match else "N/A"
    return name, email

def extract_detected_skills(text_corpus):
    skill_vocabulary = [
        "python", "java", "sql", "mysql", "postgresql", "mongodb", "javascript", "html", "css", 
        "react", "node", "angular", "aws", "docker", "kubernetes", "linux", "git", 
        "machine learning", "tensorflow", "pandas", "numpy", "scikit-learn", "pytorch"
    ]
    detected = []
    lowered_text = text_corpus.lower()
    for skill in skill_vocabulary:
        if re.search(r"\b" + re.escape(skill) + r"\b", lowered_text):
            detected.append(skill.title())
    return detected

def calculate_resume_score(text_corpus):
    score = 0
    feedback = []
    headers = {
        "Objective/Summary": ["objective", "summary", "profile"],
        "Projects": ["projects", "academic projects"],
        "Achievements": ["achievements", "awards"],
        "Hobbies": ["hobbies", "interests"],
        "Declaration": ["declaration"]
    }
    lowered_corpus = text_corpus.lower()
    for section, keywords in headers.items():
        if any(kw in lowered_corpus for kw in keywords):
            score += 20
        else:
            feedback.append(section)
    return score, feedback

def evaluate_target_domain(skills_list):
    if not skills_list: return "General/Undetermined"
    lowered_skills = [s.lower() for s in skills_list]
    domains = {
        "Data Science": ["python", "machine learning", "tensorflow", "pandas", "sql", "scikit-learn", "numpy"],
        "Web Dev": ["html", "css", "javascript", "react", "node", "angular"],
        "Cloud/DevOps": ["aws", "docker", "kubernetes", "linux", "git"]
    }
    max_matches, predicted_domain = 0, "General/Undetermined"
    for domain, keywords in domains.items():
        matches = sum(1 for kw in keywords if kw in lowered_skills)
        if matches > max_matches:
            max_matches = matches
            predicted_domain = domain
    return predicted_domain

def compute_job_match(resume_text, job_desc):
    if not job_desc.strip(): return 0
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return int(np.floor(sim[0][0] * 100))

def extract_skills_gap(resume_skills, job_desc):
    common_tech = ["python", "java", "sql", "react", "aws", "git", "machine learning", "html", "css", "javascript", "node"]
    lowered_jd = job_desc.lower()
    lowered_resume = [s.lower() for s in resume_skills]
    required = [kw for kw in common_tech if re.search(r"\b" + kw + r"\b", lowered_jd)]
    return [sk.title() for sk in required if sk not in lowered_resume]

with st.sidebar:
    st.title("SRA Engine")
    st.markdown("Automated Talent Discovery")
    st.divider()
    app_mode = st.radio("System Navigation", ["User Portal", "Admin Dashboard"])
    st.divider()
    st.caption("Powered by Natural Language Processing & TiDB Cloud Serverless Architecture.")

if app_mode == "User Portal":
    st.markdown("<h1>Smart Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("Upload candidate profiles and define target requirements to instantly extract skills, calculate formatting scores, and compute technical alignment.")
    st.divider()
    
    col_input1, col_input2 = st.columns([1, 1.5])
    
    with col_input1:
        st.subheader("Upload Candidates")
        uploaded_files = st.file_uploader("Select PDF Resumes", type=["pdf"], accept_multiple_files=True)
    
    with col_input2:
        st.subheader("Job Parameters")
        job_desc = st.text_area("Paste target Job Description here (Optional):", height=150)

    if uploaded_files:
        st.divider()
        st.subheader("Processing Results")
        with st.spinner(f"Running NLP extraction on {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                temp_filename = f"temp_{uploaded_file.name}"
                
                with open(temp_filename, "wb") as f: 
                    f.write(file_bytes)
                try:
                    raw_text = extract_text(temp_filename)
                except Exception:
                    raw_text = ""
                finally:
                    if os.path.exists(temp_filename): 
                        os.remove(temp_filename)

                if not raw_text.strip():
                    continue

                fallback_name = os.path.splitext(uploaded_file.name)[0].replace('_', ' ').replace('-', ' ').title()
                c_name, c_email = extract_contact_info(raw_text, default_name=fallback_name)
                c_skills = extract_detected_skills(raw_text)
                
                r_score, missing = calculate_resume_score(raw_text)
                domain = evaluate_target_domain(c_skills)
                match_pct = compute_job_match(raw_text, job_desc)
                gaps = extract_skills_gap(c_skills, job_desc)

                with st.expander(f"Candidate: {c_name}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Email Contact:** `{c_email}`")
                        st.markdown(f"**Predicted Domain:** {domain}")
                        st.markdown(f"**Extracted Skills:** {', '.join(c_skills) if c_skills else 'None detected'}")
                        if job_desc and gaps:
                            st.error(f"Skills Gap: {', '.join(gaps)}")
                    with col2:
                        st.metric("Format Score", f"{r_score}/100")
                        if job_desc: 
                            st.metric("Technical Match", f"{match_pct}%")

                if uploaded_file.name not in st.session_state.saved_resumes:
                    conn = get_db_connection()
                    if conn:
                        try:
                            with conn.cursor() as cursor:
                                cursor.execute("""
                                    INSERT INTO user_data (timestamp, name, email, resume_score, predicted_domain, skills, match_score)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                    str(c_name), str(c_email), int(r_score), str(domain), str(", ".join(c_skills)), int(match_pct)
                                ))
                            conn.commit()
                            conn.close()
                            st.session_state.saved_resumes.add(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Failed to sync {c_name}: {e}")
                            
        st.toast('Data Synced to TiDB Cloud!')
        st.balloons()
        st.success("Analysis complete! Candidate profiles have been securely logged in the cloud warehouse.")

elif app_mode == "Admin Dashboard":
    st.markdown("<h1>Recruiter Analytics</h1>", unsafe_allow_html=True)
    st.markdown("Real-time telemetry and candidate distribution synchronized from TiDB Serverless.")
    st.divider()
    
    with st.sidebar.expander("Database Utilities"):
        st.markdown("Use this utility to wipe placeholder data.")
        if st.button("Wipe Database Logs", type="primary", use_container_width=True):
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute("TRUNCATE TABLE user_data;")
                conn.commit()
                conn.close()
                st.session_state.saved_resumes.clear()
                st.success("Database records cleared successfully!")
                st.rerun()

    conn = get_db_connection()
    if conn:
        df = pd.DataFrame()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM user_data;")
                rows = cursor.fetchall()
            conn.close()
            
            if rows:
                df = pd.DataFrame(rows)
        except Exception as read_err:
            st.error(f"Failed to read from database: {read_err}")
        
        if not df.empty:
            df = df[df['name'] != 'name']
        
        if not df.empty:
            df['resume_score'] = pd.to_numeric(df['resume_score'], errors='coerce').fillna(0).astype(int)
            df['match_score'] = pd.to_numeric(df['match_score'], errors='coerce').fillna(0).astype(int)
            
            st.markdown("### Pipeline Metrics")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Processed Applicants", len(df), "Profiles")
            kpi2.metric("Average Formatting Score", f"{int(df['resume_score'].mean())}/100", "Quality Index")
            kpi3.metric("Highest Technical Match", f"{df['match_score'].max()}%", "Top Candidate")
            
            st.markdown("---")
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.markdown("#### Talent Distribution by Domain")
                domain_counts = df["predicted_domain"].value_counts().reset_index()
                domain_counts.columns = ["Domain", "Count"]
                
                fig_pie = px.pie(domain_counts, values="Count", names="Domain", hole=0.55, 
                                 color_discrete_sequence=px.colors.sequential.Plasma)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                      font=dict(color="white"), margin=dict(t=20, b=20, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_ch2:
                st.markdown("#### Technical Match Probability")
                
                fig_hist = px.histogram(df, x="match_score", nbins=15, 
                                        color_discrete_sequence=["#FF4B4B"], opacity=0.8)
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                       font=dict(color="white"), margin=dict(t=20, b=20, l=0, r=0),
                                       xaxis_title="Match Percentage", yaxis_title="Candidate Count")
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### Unified Candidate Data Log")
            st.dataframe(
                df[['timestamp', 'name', 'email', 'predicted_domain', 'resume_score', 'match_score', 'skills']], 
                use_container_width=True, hide_index=True
            )
        else:
            st.info("The system database warehouse is currently empty. Go ahead and analyze an actual resume file first!")
    else:
        st.warning("Database not connected. Ensure your .streamlit/secrets.toml values match your TiDB dashboard settings.")