import re
import PyPDF2
import docx
import io
import streamlit as st
from config.settings import CONFIDENCE_THRESHOLDS


def extract_text_from_resume(uploaded_file):
    """Extract text from PDF or DOCX resume"""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        else:
            raise ValueError("Unsupported file format")
        return text
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        return ""

        
def analyze_resume_consistency(resume_text, candidate_info):
    """
    Analyze resume for consistency with provided information
    Returns: (consistency_score, findings)
    """
    findings = []
    consistency_score = 1.0  # Start with perfect score
    
    # Check years of experience
    experience_patterns = [
        r'(\d+)[\+]?\s*(?:years?|yrs?).+?experience',
        r'experience.+?(\d+)[\+]?\s*(?:years?|yrs?)',
        r'(\d{4})\s*-\s*(?:present|current|now|2024)',  # Add more year patterns
    ]
    
    claimed_years = candidate_info.get("Years of Experience", 0)
    years_found = []
    
    # Extract years from dates
    for pattern in experience_patterns:
        matches = re.finditer(pattern, resume_text.lower())
        for match in matches:
            if match.group(1).isdigit():
                if len(match.group(1)) == 4:  # It's a year
                    years = 2024 - int(match.group(1))
                    years_found.append(years)
                else:
                    years_found.append(int(match.group(1)))
    
    if years_found:
        max_years = max(years_found)
        if abs(max_years - claimed_years) > 2:
            consistency_score += CONFIDENCE_THRESHOLDS['experience_mismatch_penalty']
            findings.append(f"Experience discrepancy: Claimed {claimed_years} years, Resume suggests {max_years} years")
    
    # Check skills match
    claimed_skills = set(skill.lower() for skill in candidate_info.get("Tech Stack", []))
    found_skills = set()
    
    # Build comprehensive tech stack regex
    tech_keywords = {
        'languages': r'python|java|javascript|c\+\+|ruby|php|swift|kotlin|go|rust',
        'frameworks': r'django|flask|spring|react|angular|vue|express|rails|laravel',
        'databases': r'sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra',
        'tools': r'git|docker|kubernetes|jenkins|aws|azure|gcp|terraform|ansible'
    }
    
    for category, pattern in tech_keywords.items():
        matches = re.finditer(pattern, resume_text.lower())
        for match in matches:
            found_skills.add(match.group())
    
    # Compare skills
    missing_skills = claimed_skills - found_skills
    if missing_skills:
        penalty = len(missing_skills) * CONFIDENCE_THRESHOLDS['skill_mismatch_penalty']
        consistency_score += penalty
        findings.append(f"Skills mentioned but not found in resume: {', '.join(missing_skills)}")
    
    # Check position alignment
    desired_position = candidate_info.get("Desired Position", "").lower()
    position_words = set(desired_position.split())
    position_match = any(word in resume_text.lower() for word in position_words if len(word) > 3)
    
    if not position_match:
        consistency_score += CONFIDENCE_THRESHOLDS['resume_mismatch_penalty']
        findings.append("Desired position not aligned with resume content")
    else:
        consistency_score += CONFIDENCE_THRESHOLDS['resume_match_bonus']
    
    # Normalize final score
    consistency_score = max(0.0, min(consistency_score, 1.0))
    
    return consistency_score, findings