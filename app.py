import os
import re
import json
import datetime
import sqlite3
import fitz
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# --- Securely Configure the Gemini API ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set. AI suggestions will be disabled.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# --- Load NLP Models ---
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

app = Flask(__name__)

# --- Database functions ---
def init_db():
    conn = sqlite3.connect('analyzer.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, score INTEGER NOT NULL,
            matched_count INTEGER NOT NULL, missing_count INTEGER NOT NULL,
            jd_preview TEXT NOT NULL, analyzed_at TIMESTAMP NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# === Master Skill Set & Synonyms ===
SKILLS_SET = {
    "python", "sql", "aws", "power bi", "machine learning", "streamlit", "folium", "api", "iot", "firebase", "android app", "rag",
    "java", "c++", "c#", "javascript", "typescript", "mysql", "git", "github", "docker", "ec2", "s3", "data pipelines",
    "kubernetes", "azure", "gcp", "react", "angular", "vue", "node.js", "django", "flask",
    "data analysis", "data science", "tensorflow", "pytorch", "keras", "scikit-learn",
    "rest", "graphql", "html", "css", "agile", "scrum", "jira", "project management",
    "pyspark", "spark", "hadoop", "big data", "tableau", "data visualization", "etl",
    "natural language processing", "nlp", "nltk", "computer vision", "deep learning", "linux",
    "mongodb", "nosql", "hdfs", "mapreduce", "hive", "hbase", "matplotlib", "seaborn", "pandas", "numpy",
    "databricks", "blob storage", "app services", "advance analytics", "data collection",
    "dbms", "r programming", "object oriented programming"
}
SKILL_SYNONYMS = {
    "r programming": ["r", "r language"], "object oriented programming": ["oop", "object oriented"], "sql": ["sql", "dbms", "database"],
    "etl": ["etl", "data pipeline", "data pipelines", "data wrangling"], "hadoop": ["hadoop", "big data technologies"],
    "spark": ["spark", "pyspark", "apache spark", "big data technologies"], "aws": ["aws", "ec2", "s3", "cloud computing"],
    "hive": ["hive", "big data technologies"], "hbase": ["hbase", "big data technologies"], "hdfs": ["hdfs", "big data technologies"],
    "mapreduce": ["mapreduce", "big data technologies"], "nosql": ["nosql", "firebase", "mongodb"],
    "data visualization": ["data visualization", "power bi", "matplotlib", "seaborn"],
    "advance analytics": ["advance analytics", "statistical methods", "predictive modeling"]
}
KNOWN_LOCATIONS = {"mumbai", "india", "kharghar", "navi mumbai", "thane", "maharashtra", "ulwe", "pune", "chinchwad"}
ORG_EXCLUSION_KEYWORDS = {
    "board", "committee", "course", "fest", "program", "programming", "lead", "team", "services", "reporting",
    "analysis", "engineering", "functions", "address", "analytics", "development", "data", "big data", "machine",
    "indian", "society", "mandal", "club"
}

# === Helper Functions ===
def clean_text(text):
    text = text.replace('##', '')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n+', '\n', text)
    return text

def normalize_skills(text):
    found = set()
    text_lower = text.lower().replace('-', ' ')
    for skill in SKILLS_SET:
        if re.search(rf"\b{re.escape(skill)}\b", text_lower):
            found.add(skill)
    for norm, aliases in SKILL_SYNONYMS.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                found.add(norm)
    return found

def extract_weighted_jd_skills(jd_text):
    jd_lower = jd_text.lower()
    weighted_skills = {}
    required_phrases = ["required", "must have", "strong experience", "essential"]
    preferred_phrases = ["preferred", "plus", "nice to have", "bonus"]
    normalized_jd_skills = normalize_skills(jd_text)
    for skill in normalized_jd_skills:
        weight = 2
        try:
            for match in re.finditer(rf"\b{re.escape(skill)}\b", jd_lower):
                context_window = jd_lower[max(0, match.start() - 50):match.start()]
                if any(phrase in context_window for phrase in required_phrases):
                    weight = 3; break
                elif any(phrase in context_window for phrase in preferred_phrases):
                    weight = 1; break
        except re.error:
            continue
        if skill not in weighted_skills or weight > weighted_skills[skill]:
            weighted_skills[skill] = weight
    return weighted_skills

def analyze_experience(text):
    patterns = [r'(\d+\s*\+\s*years?)', r'(\d+\s*-\s*\d+\s*years?)', r'(\d+\s+years?)', r'(several\s+years?)']
    found_experience = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            found_experience.append(match.strip())
    return list(set(found_experience))

# === Gemini-Powered Suggestion Functions ===
def generate_ai_suggestion(missing_skills, jd_text):
    if not genai or not os.environ.get('GOOGLE_API_KEY'): return ["[AI Suggestion feature is not configured.]"]
    if not missing_skills: return []
    job_title_match = re.search(r'job title:\s*(.*)', jd_text, re.IGNORECASE)
    job_title = job_title_match.group(1).strip() if job_title_match else "the role"
    top_missing = sorted(list(missing_skills))[:4]
    prompt = f"As a career coach for a candidate applying for '{job_title}' who is missing skills like {', '.join(top_missing)}, provide three distinct, actionable resume update suggestions in a numbered list."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        raw_suggestions = response.text.strip().split('\n')
        return [re.sub(r'^\d+\.\s*', '', s).strip() for s in raw_suggestions if s.strip()] or []
    except Exception as e:
        print(f"Gemini suggestion generation failed: {e}"); return []

def analyze_action_verbs_with_gemini(resume_text):
    if not genai or not os.environ.get('GOOGLE_API_KEY'): return []
    projects_section_match = re.search(r'Academic Projects\s*-*([\s\S]*)', resume_text, re.IGNORECASE)
    text_to_analyze = projects_section_match.group(1) if projects_section_match else resume_text[-1500:]
    if not text_to_analyze.strip(): return []
    prompt = f"""Analyze the following resume text. Identify up to 3 weak phrases (e.g., "worked on") and suggest a specific rewrite starting with a strong action verb. Return ONLY a JSON object like: {{"rewrites": [{{"original_phrase": "...", "suggested_rewrite": "..."}}]}} --- TEXT: {text_to_analyze}"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        analysis_json = json.loads(cleaned_response)
        suggestions = []
        if analysis_json.get("rewrites"):
            for item in analysis_json["rewrites"]:
                suggestions.append(f"Instead of '{item.get('original_phrase')}', consider a more direct phrasing like: **'{item.get('suggested_rewrite')}'**.")
        return suggestions
    except (json.JSONDecodeError, Exception) as e:
        print(f"Gemini action verb analysis failed: {e}"); return []

# === Main Analyzer ===
def analyze_resume(resume_text, jd_text):
    cleaned_resume_text = clean_text(resume_text)
    resume_skills = normalize_skills(cleaned_resume_text)
    weighted_jd_skills = extract_weighted_jd_skills(jd_text)
    matched_skills = resume_skills.intersection(weighted_jd_skills.keys())
    missing_skills = set(weighted_jd_skills.keys()) - resume_skills
    achieved_score = sum(weighted_jd_skills[skill] for skill in matched_skills)
    total_possible_score = sum(weighted_jd_skills.values())
    score = round((achieved_score / total_possible_score) * 100) if total_possible_score > 0 else 100
    resume_experience = analyze_experience(cleaned_resume_text)
    jd_experience = analyze_experience(jd_text)
    technical_suggestions = generate_ai_suggestion(missing_skills, jd_text)
    rewrite_suggestions = analyze_action_verbs_with_gemini(cleaned_resume_text)
    if missing_skills:
        sorted_missing = sorted(list(missing_skills), key=lambda s: weighted_jd_skills.get(s, 0), reverse=True)
        technical_suggestions.append(f"Prioritize adding these key skills from the job description: {', '.join(sorted_missing[:4])}.")
    if jd_experience and not resume_experience:
        technical_suggestions.append(f"The job description mentions '{jd_experience[0]}'. Make sure your resume clearly states your total years of experience.")
    
    final_locations = {loc for loc in KNOWN_LOCATIONS if loc in cleaned_resume_text.lower()}
    person_name_parts = []
    for line in cleaned_resume_text.split('\n'):
        if line.strip() and len(line.strip().split()) < 4:
            person_name_parts = line.strip().lower().split(); break
    ner_results = ner_pipeline(cleaned_resume_text)
    potential_orgs = set()
    for entity in ner_results:
        if entity['score'] < 0.65: continue
        entity_text = entity['word'].strip(); entity_text_lower = entity_text.lower()
        if entity['entity_group'] == 'ORG':
            if any([entity_text_lower in SKILLS_SET, entity_text_lower in KNOWN_LOCATIONS,
                    any(part in entity_text_lower for part in person_name_parts if len(part) > 3),
                    any(keyword in entity_text_lower for keyword in ORG_EXCLUSION_KEYWORDS), len(entity_text) < 4]):
                continue
            potential_orgs.add(entity_text)
    final_orgs = set()
    for org in sorted(list(potential_orgs), key=len, reverse=True):
        if not any(org in other_org and org != other_org for other_org in final_orgs):
            final_orgs.add(org)
    return {
        "score": score, "matched_keywords": sorted(list(matched_skills)), "missing_keywords": sorted(list(missing_skills)),
        "suggestions": {"technical": technical_suggestions, "rewrites": rewrite_suggestions},
        "resume_text": cleaned_resume_text, "jd_text": jd_text,
        "experience": {"resume": resume_experience, "job_description": jd_experience},
        "entities": {"skills": sorted([s.title() for s in resume_skills]), "organizations": sorted(list(final_orgs)), "locations": sorted([l.title() for l in final_locations])}
    }

# === Flask Routes ===
@app.route('/')
def index():
    # The history is no longer fetched and passed to the main page
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_text = ""
    if 'resume-file' in request.files and request.files['resume-file'].filename != '':
        file = request.files['resume-file']
        if file.filename.endswith('.pdf'):
            try:
                doc = fitz.open(stream=file.read(), filetype="pdf")
                resume_text = "".join(page.get_text() for page in doc)
                doc.close()
            except Exception as e:
                return jsonify({"error": f"Error processing PDF: {e}"}), 400
        else:
            return jsonify({"error": "Please upload a PDF."}), 400
    elif 'resume' in request.form:
        resume_text = request.form.get('resume', '')
    
    jd_text = request.form.get('jd', '')
    if not resume_text or not jd_text:
        return jsonify({"error": "Resume or Job Description missing."}), 400

    analysis_result = analyze_resume(resume_text, jd_text)

    # Save every analysis to the database in the background
    try:
        conn = sqlite3.connect('analyzer.db')
        cursor = conn.cursor()
        jd_preview = (jd_text[:75] + '...') if len(jd_text) > 75 else jd_text
        cursor.execute('INSERT INTO analyses (score, matched_count, missing_count, jd_preview, analyzed_at) VALUES (?, ?, ?, ?, ?)',
                       (analysis_result['score'], len(analysis_result['matched_keywords']), len(analysis_result['missing_keywords']), jd_preview, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database write error: {e}")

    return jsonify(analysis_result)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=8080)