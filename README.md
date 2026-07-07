# 🎯 Smart Resume Analyzer (SRA Engine)

A cloud-connected, NLP-powered application designed to automate resume screening, calculate technical alignment, and provide recruiters with real-time talent analytics. 

This project bridges the gap between raw applicant data and actionable hiring metrics using machine learning and a serverless cloud database architecture.

## ✨ Features

* **🧠 NLP Job Matching:** Uses TF-IDF vectorization and Cosine Similarity to calculate a precise "Match Percentage" between candidate resumes and target job descriptions.
* **📝 Smart Parsing:** Extracts candidate contact information, predicts their professional domain, and isolates technical skills from raw PDF documents.
* **📊 Recruiter Dashboard:** A real-time analytics portal featuring interactive donut charts and histograms to visualize talent distribution and scoring metrics.
* **☁️ Cloud Synchronized:** All candidate data is securely stored and retrieved in real-time using TiDB Cloud Serverless architecture.
* **⚡ Skill Gap Analysis:** Automatically cross-references extracted candidate skills against common industry requirements found in the job description to highlight missing qualifications.

## 🛠️ Tech Stack

* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Database:** [TiDB Cloud Serverless](https://en.pingcap.com/tidb-cloud/) (MySQL compatible)
* **NLP & Machine Learning:** `scikit-learn` (TF-IDF, Cosine Similarity)
* **Data Visualization:** `plotly` & `pandas`
* **PDF Processing:** `pdfminer.six`

## 🚀 Local Installation & Setup

If you want to run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/smart-resume-analyzer.git](https://github.com/your-username/smart-resume-analyzer.git)
   cd smart-resume-analyzer
