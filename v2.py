# import streamlit as st
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain_groq import ChatGroq
# from datetime import datetime
# import os
# import json
# import re
# import PyPDF2
# import docx
# import io
# from models.llm_manager import LLMManager


import streamlit as st
from config.settings import CONFIDENCE_THRESHOLDS, CONVERSATION_MEMORY_LENGTH
from utils.validators import validate_email, validate_phone, validate_tech_stack
from utils.resume_processing import extract_text_from_resume, analyze_resume_consistency
from components.sidebar import render_sidebar
from components.progress import create_progress_container, update_assessment_progress
from assessment.question_generation import generate_technical_questions, generate_focused_question, similar_questions
from assessment.evaluation import (
    evaluate_answer_with_llm,
    fallback_evaluation,
    generate_detailed_feedback_with_llm,
    generate_final_recommendation_with_llm,
    generate_fallback_recommendation,
    assess_confidence_level,
    determine_focus_areas,
    extract_technical_terms
)
from reporting.report_generator import generate_report
from models.llm_manager import determine_optimal_persona, get_persona_prompt, LLMManager
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from datetime import datetime
import os
import json
import re
import PyPDF2
import docx
import io


# Initialize Streamlit page configuration
st.set_page_config(
    page_title='TalentScout Hiring Assistant 🤖',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded'
)



# Load environment variables securely
if 'GROQ_API_KEY' not in st.secrets:
    st.error('Please set the GROQ_API_KEY in your Streamlit secrets.')
    st.stop()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
CONVERSATION_MEMORY_LENGTH = 10
# Initialize session state variables
def initialize_session_state():
    session_vars = {
        'chat_history': [],
        'total_messages': 0,
        'start_time': None,
        'candidate_info': {},
        'current_question': None,
        'assessment_completed': False,
        'answers': {},
        'evaluation_scores': {},
        'recommendation': None,
        'technical_questions': [],
        'current_question_index': 0,
        'current_answer': '',  # Add this new state variable
        'questions_asked': 0  # Add the missing questions_asked variable
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default


# # Input validation functions
# def validate_email(email):
#     pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#     return bool(re.match(pattern, email))

# def validate_phone(phone):
#     # Basic phone validation - can be adjusted based on your needs
#     pattern = r'^\+?1?\d{9,15}$'
#     return bool(re.match(pattern, phone))

# def validate_tech_stack(tech_stack):
#     techs = [tech.strip() for tech in tech_stack.split(',') if tech.strip()]
#     return len(techs) > 0


# def create_assessment_guidelines():
#     """Create consistent assessment guidelines for sidebar"""
#     guidelines = {
#         "steps": [
#             "📋 Initial Information Collection",
#             "📎 Resume Analysis & Verification",
#             "🔍 Technical Skills Assessment",
#             "📊 Performance Evaluation",
#             "📈 Final Report Generation"
#         ],
#         "rules": [
#             "Answer each question thoroughly and precisely",
#             "Take time to structure your responses clearly",
#             "Focus on practical experience and examples",
#             "Be honest about knowledge limitations",
#             "Maintain professional communication"
#         ]
#     }
#     return guidelines


# def generate_motivation_message(resume_analysis_results):
#     """Generate personalized motivation based on resume analysis"""
#     consistency_score = resume_analysis_results.get('consistency_score', 0)
#     strengths = resume_analysis_results.get('strengths', [])
    
#     # Craft personalized motivation
#     if consistency_score >= 0.8 and strengths:
#         message = f"Your experience in {strengths[0]} stands out. Let's showcase your expertise!"
#     elif consistency_score >= 0.6:
#         message = "Your background shows promise. This assessment will highlight your potential."
#     else:
#         message = "Every question is an opportunity to demonstrate your capabilities!"
    
#     return message

# def render_sidebar(stage, resume_analysis=None):
#     """Render consistent sidebar content across all stages"""
#     with st.sidebar:
#         st.header('Assessment Guide 📚')
        
#         # Display current stage
#         stages = {
#             'info': '📋 Information Collection',
#             'assessment': '🔍 Technical Assessment',
#             'report': '📈 Final Report'
#         }
#         current_stage = stages.get(stage, '')
#         st.subheader(f"Current Stage: {current_stage}")
        
#         # Display guidelines
#         guidelines = create_assessment_guidelines()
        
#         with st.expander("📝 Assessment Steps", expanded=True):
#             for idx, step in enumerate(guidelines['steps'], 1):
#                 if stages[stage] in step:
#                     st.markdown(f"**→ {idx}. {step}**")
#                 else:
#                     st.markdown(f"{idx}. {step}")
        
#         with st.expander("📋 Assessment Rules", expanded=True):
#             for rule in guidelines['rules']:
#                 st.markdown(f"• {rule}")
        
#         # Display motivation if resume has been analyzed
#         if resume_analysis and 'consistency_score' in resume_analysis:
#             st.markdown("---")
#             st.subheader("💫 Your Assessment Journey")
#             motivation = generate_motivation_message(resume_analysis)
#             st.markdown(f"*{motivation}*")
        
#         # Clear session button at bottom
#         st.markdown("---")
#         if st.button('Reset Assessment 🔄'):
#             for key in list(st.session_state.keys()):
#                 del st.session_state[key]
#             st.rerun()


# def extract_text_from_resume(uploaded_file):
#     """Extract text from PDF or DOCX resume"""
#     text = ""
#     try:
#         if uploaded_file.type == "application/pdf":
#             pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#             doc = docx.Document(io.BytesIO(uploaded_file.read()))
#             for paragraph in doc.paragraphs:
#                 text += paragraph.text + "\n"
#         else:
#             raise ValueError("Unsupported file format")
#         return text
#     except Exception as e:
#         st.error(f"Error processing resume: {str(e)}")
#         return ""

# def analyze_resume_consistency(resume_text, candidate_info):
#     """
#     Analyze resume for consistency with provided information
#     Returns: (consistency_score, findings)
#     """
#     findings = []
#     consistency_score = 1.0  # Start with perfect score
    
#     # Check years of experience
#     experience_patterns = [
#         r'(\d+)[\+]?\s*(?:years?|yrs?).+?experience',
#         r'experience.+?(\d+)[\+]?\s*(?:years?|yrs?)',
#         r'(\d{4})\s*-\s*(?:present|current|now|2024)',  # Add more year patterns
#     ]
    
#     claimed_years = candidate_info.get("Years of Experience", 0)
#     years_found = []
    
#     # Extract years from dates
#     for pattern in experience_patterns:
#         matches = re.finditer(pattern, resume_text.lower())
#         for match in matches:
#             if match.group(1).isdigit():
#                 if len(match.group(1)) == 4:  # It's a year
#                     years = 2024 - int(match.group(1))
#                     years_found.append(years)
#                 else:
#                     years_found.append(int(match.group(1)))
    
#     if years_found:
#         max_years = max(years_found)
#         if abs(max_years - claimed_years) > 2:
#             consistency_score += CONFIDENCE_THRESHOLDS['experience_mismatch_penalty']
#             findings.append(f"Experience discrepancy: Claimed {claimed_years} years, Resume suggests {max_years} years")
    
#     # Check skills match
#     claimed_skills = set(skill.lower() for skill in candidate_info.get("Tech Stack", []))
#     found_skills = set()
    
#     # Build comprehensive tech stack regex
#     tech_keywords = {
#         'languages': r'python|java|javascript|c\+\+|ruby|php|swift|kotlin|go|rust',
#         'frameworks': r'django|flask|spring|react|angular|vue|express|rails|laravel',
#         'databases': r'sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra',
#         'tools': r'git|docker|kubernetes|jenkins|aws|azure|gcp|terraform|ansible'
#     }
    
#     for category, pattern in tech_keywords.items():
#         matches = re.finditer(pattern, resume_text.lower())
#         for match in matches:
#             found_skills.add(match.group())
    
#     # Compare skills
#     missing_skills = claimed_skills - found_skills
#     if missing_skills:
#         penalty = len(missing_skills) * CONFIDENCE_THRESHOLDS['skill_mismatch_penalty']
#         consistency_score += penalty
#         findings.append(f"Skills mentioned but not found in resume: {', '.join(missing_skills)}")
    
#     # Check position alignment
#     desired_position = candidate_info.get("Desired Position", "").lower()
#     position_words = set(desired_position.split())
#     position_match = any(word in resume_text.lower() for word in position_words if len(word) > 3)
    
#     if not position_match:
#         consistency_score += CONFIDENCE_THRESHOLDS['resume_mismatch_penalty']
#         findings.append("Desired position not aligned with resume content")
#     else:
#         consistency_score += CONFIDENCE_THRESHOLDS['resume_match_bonus']
    
#     # Normalize final score
#     consistency_score = max(0.0, min(consistency_score, 1.0))
    
#     return consistency_score, findings



# # Define persona-based prompt templates
# def get_persona_prompt(persona):
#     personas = {
#     'Default': ChatPromptTemplate.from_messages([
#         ("system", """You are a friendly and professional hiring assistant. 
#                      Your role is to conduct preliminary technical screenings for candidates. 
#                      Focus on gathering essential details, maintaining a conversational tone, 
#                      and assessing both technical knowledge and problem-solving abilities. 
#                      Provide constructive feedback without overwhelming the candidate."""),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),
#     ]),
    
#     'Expert': ChatPromptTemplate.from_messages([
#         ("system", """You are a highly experienced technical hiring manager. 
#                      Your job is to assess candidates thoroughly on:
#                      - Technical accuracy
#                      - Problem-solving strategies
#                      - Code quality and optimization
#                      - System design and scalability
#                      Start with foundational questions, then dive into advanced topics 
#                      and edge cases. Offer precise, actionable feedback based on the 
#                      candidate's responses, highlighting strengths and improvement areas."""),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),
#     ]),
    
#     'Creative': ChatPromptTemplate.from_messages([
#         ("system", """You are an engaging and innovative interviewer who evaluates 
#                      candidates through real-world scenarios and practical challenges. 
#                      Assess:
#                      - Creative problem-solving
#                      - Adaptability to unique scenarios
#                      - Application of technical knowledge
#                      - Clear and concise communication
#                      Use situational questions and collaborative problem-solving exercises 
#                      to encourage critical thinking."""),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),
#     ]),

#     'Analytical': ChatPromptTemplate.from_messages([
#         ("system", """You are a data-driven and analytical evaluator. 
#                      Your focus is on assessing logical reasoning and analytical skills 
#                      alongside technical expertise. Start with short and specific 
#                      questions, progressing to scenarios that require deeper analysis. 
#                      Evaluate based on:
#                      - Clarity in logic
#                      - Efficiency in problem-solving
#                      - Ability to break down complex problems into manageable steps."""),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),
#     ]),
# }
#     return personas.get(persona, personas['Default'])

# # Function to generate technical questions
# def generate_technical_questions(tech_stack, conversation):
#     prompt = f"""
#     Based on the tech stack: {tech_stack}, generate 5 questions with increasing difficulty:

#     1. Start with a basic concept/definition question (easy, short answer)
#     2. Progress to fundamentals application (moderate, brief explanation)
#     3. Add a practical scenario (moderate, focused solution)
#     4. Include problem-solving (challenging but specific)
#     5. End with advanced concepts (if needed based on previous answers)

#     Rules:
#     - First 2 questions should be answerable in 1-2 sentences
#     - Questions 3-4 should need 3-4 sentences max
#     - Keep questions focused and specific
#     - Avoid asking for code implementations
#     - Use this format: "Question N: [The question text]"

#     Example progression:
#     Question 1: What is [basic concept] in {tech_stack}?
#     Question 2: How does [fundamental feature] work in {tech_stack}?
#     Question 3: In a simple web app, how would you handle [specific scenario]?
#     Question 4: What approach would you take to solve [specific problem]?
#     Question 5: Explain the trade-offs between [advanced concept A] and [advanced concept B].

#     Generate questions that are concise and clear.
#     """
   
#     try:
#         response = conversation.predict(input=prompt)
        
#         questions = []
#         for line in response.splitlines():
#             if line.strip().startswith('Question'):
#                 questions.append(line.strip())
        
#         return questions[:5]  # Ensure we only return 5 questions
#     except Exception as e:
#         st.error(f"Error generating questions: {str(e)}")
#         return ["Error generating questions. Please try again."]







# # def create_evaluation_llm():
# #     """Create a separate LLM instance for evaluation"""
# #     return ChatGroq(
# #         api_key=st.secrets["GROQ_API_KEY"],
# #         model_name='llama-3.3-70b-versatile',
# #         temperature=0.5,  # Lower temperature for more consistent evaluation
# #         max_tokens=4028
# #     )

# def evaluate_answer_with_llm(question, answer, tech_stack):
#     """Evaluate answer using LLM with improved response handling"""
#     evaluation_llm = LLMManager.get_llm('evaluation')
    
#     prompt = f"""You are an expert technical interviewer evaluating a candidate's response. You must return your evaluation in the exact JSON format specified below.

# Question: {question}
# Candidate's Answer: {answer}
# Relevant Technologies: {', '.join(tech_stack)}

# Evaluate the answer across these dimensions:
# 1. Technical Accuracy
# 2. Completeness
# 3. Clarity
# 4. Best Practices

# Your response must be in this exact JSON format with no additional text before or after:
# {{
#     "technical_accuracy": {{
#         "score": <number between 0-100>,
#         "feedback": "<specific feedback>"
#     }},
#     "completeness": {{
#         "score": <number between 0-100>,
#         "feedback": "<specific feedback>"
#     }},
#     "clarity": {{
#         "score": <number between 0-100>,
#         "feedback": "<specific feedback>"
#     }},
#     "best_practices": {{
#         "score": <number between 0-100>,
#         "feedback": "<specific feedback>"
#     }},
#     "overall_feedback": "<summarizing feedback>"
# }}

# Remember: Return only valid JSON, no other text."""
    
#     try:
#         # Get LLM response
#         response = evaluation_llm.predict(prompt)
        
#         # Clean the response - remove any potential markdown formatting or extra text
#         response = response.strip()
#         if response.startswith('```json'):
#             response = response[7:]
#         if response.endswith('```'):
#             response = response[:-3]
#         response = response.strip()
        
#         # Try to parse the JSON
#         try:
#             evaluation = json.loads(response)
#         except json.JSONDecodeError:
#             # If JSON parsing fails, create a default structured response
#             st.warning("AI response formatting issue. Using simplified evaluation.")
#             return fallback_evaluation(answer)
        
#         # Calculate normalized scores (0-1 range)
#         scores = {
#             'technical_accuracy': float(evaluation['technical_accuracy']['score']) / 100,
#             'completeness': float(evaluation['completeness']['score']) / 100,
#             'clarity': float(evaluation['clarity']['score']) / 100,
#             'best_practices': float(evaluation['best_practices']['score']) / 100
#         }
        
#         # Collect feedback
#         feedback = [
#             f"Technical Accuracy: {evaluation['technical_accuracy']['feedback']}",
#             f"Completeness: {evaluation['completeness']['feedback']}",
#             f"Clarity: {evaluation['clarity']['feedback']}",
#             f"Best Practices: {evaluation['best_practices']['feedback']}",
#             f"\nOverall: {evaluation['overall_feedback']}"
#         ]
        
#         # Calculate final score as weighted average
#         weights = {
#             'technical_accuracy': 0.4,
#             'completeness': 0.2,
#             'clarity': 0.2,
#             'best_practices': 0.2
#         }
#         final_score = sum(scores[k] * weights[k] for k in weights)
        
#         return final_score, feedback
        
#     except Exception as e:
#         st.warning(f"Using fallback evaluation due to: {str(e)}")
#         return fallback_evaluation(answer)

# def fallback_evaluation(answer):
#     """Provide a basic evaluation when LLM evaluation fails"""
#     # Basic scoring based on answer length and complexity
#     length_score = min(len(answer.split()) / 100, 1.0)  # Normalize based on word count
    
#     # Basic complexity score based on technical term usage
#     technical_terms = ['function', 'class', 'method', 'algorithm', 'complexity', 'performance', 'optimization']
#     tech_term_count = sum(1 for term in technical_terms if term.lower() in answer.lower())
#     complexity_score = min(tech_term_count / 5, 1.0)
    
#     # Calculate final score
#     final_score = (length_score + complexity_score) / 2
    
#     # Generate basic feedback
#     feedback = [
#         f"Answer length: {'Good' if length_score > 0.7 else 'Could be more detailed'}",
#         f"Technical depth: {'Good' if complexity_score > 0.7 else 'Could include more technical details'}",
#         "\nOverall: The answer has been evaluated using basic metrics. Please try submitting again for a more detailed AI evaluation."
#     ]
    
#     return final_score, feedback

# # def create_evaluation_llm():
# #     """Create a separate LLM instance for evaluation with optimized parameters"""
# #     return ChatGroq(
# #         api_key=st.secrets["GROQ_API_KEY"],
# #         model_name='llama-3.3-70b-versatile',
# #         temperature=0.4,  # Slightly higher for more detailed responses
# #         max_tokens=4028,  # Increased for longer responses
# #         top_p=0.95,
# #         presence_penalty=0.6,  # Encourage more diverse responses
# #         frequency_penalty=0.3  # Reduce repetition
# #     )  


# def generate_detailed_feedback_with_llm(answers, tech_stack):
#     """Generate comprehensive feedback using LLM"""
#     feedback_llm = LLMManager.get_llm('evaluation')
    
#     answers_summary = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items()])
    
#     prompt = f"""
#     Review these technical interview answers:
#     {answers_summary}
    
#     Technologies: {', '.join(tech_stack)}
    
#     Provide a comprehensive evaluation including:
#     1. Key strengths
#     2. Areas for improvement
#     3. Technical proficiency level
#     4. Specific recommendations
    
#     Format your response as detailed but precise and to the point paragraphs.
#     Make sure you Explore complex thoughts through introspective, analytical, and philosophical self-examination but provide simple and clear and short feedback.
#     """
    
#     try:
#         detailed_feedback = feedback_llm.predict(prompt)
#         return detailed_feedback
#     except Exception as e:
#         return f"Error generating detailed feedback: {str(e)}"

# def generate_final_recommendation_with_llm(candidate_info, answers, scores):
#     """Generate final recommendation using LLM with enhanced prompting and fallback"""
#     recommendation_llm = LLMManager.get_llm('recommendation')
    
#     # Calculate key metrics for context
#     avg_score = sum(scores.values()) / len(scores) if scores else 0
#     strengths = [q for q, s in scores.items() if s >= 0.7]
#     areas_for_improvement = [q for q, s in scores.items() if s < 0.7]
    
#     prompt = f"""As an expert technical interviewer, provide a detailed hiring recommendation. You MUST follow this specific structure in your response.

# CANDIDATE PROFILE:
# {json.dumps(candidate_info, indent=2)}

# ASSESSMENT METRICS:
# - Average Score: {avg_score * 100:.1f}%
# - Questions Answered: {len(answers)}
# - Strong Areas: {len(strengths)} questions
# - Areas Needing Improvement: {len(areas_for_improvement)} questions

# YOUR TASK:
# Provide a comprehensive recommendation following EXACTLY this format:

# 1. RECOMMENDATION: [Must be one of: "Strong Hire", "Hire", "Hold - Need More Information", "No Hire"]

# 2. JUSTIFICATION:
# - Technical Skills Assessment
# - Problem Solving Abilities
# - Communication Quality
# - Overall Fit for Role

# 3. KEY STRENGTHS:
# - [List at least 3 specific strengths]

# 4. AREAS FOR IMPROVEMENT:
# - [List at least 2 specific areas]

# 5. SUGGESTED NEXT STEPS:
# - [Provide at least 3 specific actionable steps]

# You MUST fill out all sections in detail. Keep your response professional and constructive, even for "No Hire" recommendations.
# """

#     try:
#         recommendation = recommendation_llm.predict(prompt)
        
#         # Verify if the response has all required sections
#         required_sections = [
#             "RECOMMENDATION:", 
#             "JUSTIFICATION:", 
#             "KEY STRENGTHS:", 
#             "AREAS FOR IMPROVEMENT:", 
#             "SUGGESTED NEXT STEPS:"
#         ]
        
#         if not all(section in recommendation for section in required_sections):
#             # If missing sections, try one more time with a more forceful prompt
#             return generate_fallback_recommendation(candidate_info, answers, scores)
            
#         return recommendation
        
#     except Exception as e:
#         return generate_fallback_recommendation(candidate_info, answers, scores)


# def generate_fallback_recommendation(candidate_info, answers, scores):
#     """Generate a structured recommendation when the primary method fails"""
#     avg_score = sum(scores.values()) / len(scores) if scores else 0
    
#     # Define recommendation based on average score
#     if avg_score >= 0.8:
#         hire_status = "Strong Hire"
#         confidence = "high"
#     elif avg_score >= 0.7:
#         hire_status = "Hire"
#         confidence = "moderate"
#     elif avg_score >= 0.5:
#         hire_status = "Hold - Need More Information"
#         confidence = "low"
#     else:
#         hire_status = "No Hire"
#         confidence = "moderate"

#     # Generate structured recommendation
#     recommendation = f"""
# 1. RECOMMENDATION: {hire_status}

# 2. JUSTIFICATION:
# - Technical Skills Assessment: Candidate demonstrated {'strong' if avg_score >= 0.7 else 'moderate' if avg_score >= 0.5 else 'insufficient'} technical knowledge
# - Problem Solving Abilities: {'Effectively' if avg_score >= 0.7 else 'Adequately' if avg_score >= 0.5 else 'Insufficiently'} solved presented challenges
# - Communication Quality: Responses were {'clear and well-structured' if avg_score >= 0.7 else 'adequate' if avg_score >= 0.5 else 'needing improvement'}
# - Overall Fit for Role: {'Strong' if avg_score >= 0.7 else 'Potential' if avg_score >= 0.5 else 'Limited'} alignment with position requirements

# 3. KEY STRENGTHS:
# - {'Demonstrated technical knowledge in ' + ', '.join(candidate_info.get('Tech Stack', ['relevant areas'])[:3])}
# - {'Strong problem-solving approach' if avg_score >= 0.7 else 'Basic understanding of concepts'}
# - {'Clear communication skills' if avg_score >= 0.6 else 'Willingness to engage with technical questions'}

# 4. AREAS FOR IMPROVEMENT:
# - {'Advanced concepts in ' + ', '.join(candidate_info.get('Tech Stack', ['relevant areas'])[:2])}
# - {'Detailed problem analysis' if avg_score < 0.8 else 'Edge case handling'}
# - {'Technical communication clarity' if avg_score < 0.7 else 'Advanced scenario handling'}

# 5. SUGGESTED NEXT STEPS:
# - {'Schedule final round interview' if hire_status == 'Strong Hire' else 'Conduct additional technical assessment' if hire_status == 'Hold - Need More Information' else 'Consider for different role/level'}
# - {'Prepare system design discussion' if avg_score >= 0.7 else 'Review fundamental concepts'}
# - {'Discuss team fit and project experience' if avg_score >= 0.6 else 'Gain more practical experience'}
# - {'Evaluate architectural knowledge' if avg_score >= 0.8 else 'Focus on core competency development'}

# Note: This recommendation is based on quantitative assessment scores and candidate profile analysis.
# """
    
#     return recommendation


# # Modified generate_report function
# def generate_report(candidate_info, answers, evaluation_scores, recommendation):
#     report_llm = LLMManager.get_llm('report')
    
#     prompt = f"""
#     Generate a comprehensive assessment report for:
    
#     Candidate: {json.dumps(candidate_info, indent=2)}
#     Evaluation Scores: {json.dumps(evaluation_scores, indent=2)}
#     Recommendation: {recommendation}
    
#     Include:
#     1. Executive summary
#     2. Technical evaluation
#     3. Key observations
#     4. Next steps
    
#     Format the response as a detailed JSON report.
#     """
    
#     try:
#         report_content = report_llm.predict(prompt)
#         report_json = json.loads(report_content)
        
#         # Add metadata
#         report_json["Report Generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         report_json["Candidate Information"] = candidate_info
#         report_json["Technical Assessment"] = {
#             question: {
#                 "Answer": answer,
#                 "Score": f"{evaluation_scores.get(question, 0) * 100:.1f}%"
#             }
#             for question, answer in answers.items()
#         }
        
#         return json.dumps(report_json, indent=4)
#     except Exception as e:
#         return json.dumps({
#             "error": f"Error generating report: {str(e)}",
#             "basic_info": {
#                 "candidate": candidate_info,
#                 "scores": evaluation_scores,
#                 "recommendation": recommendation
#             }
#         }, indent=4)


# CONFIDENCE_THRESHOLDS = {
#     'perfect_answer': 0.15,  # Increase for perfect answers
#     'good_answer': 0.08,     # Increase for good answers
#     'poor_answer': -0.12,    # Decrease for poor answers
#     'skip_penalty': -0.2,    # Heavy penalty for skips
#     'max_confidence': 0.95,  # Maximum possible confidence
#     'min_confidence': 0.0,   # Minimum possible confidence
#     'completion_threshold': 0.85,  # Confidence needed to complete
#     'skip_threshold': 3,     # Maximum allowed skips
#     'poor_answer_threshold': 4,  # Maximum allowed poor answers
#     'resume_mismatch_penalty': -0.15,  # Penalty for inconsistencies
#     'resume_match_bonus': 0.1,         # Bonus for strong matches
#     'skill_mismatch_penalty': -0.08,   # Penalty for each missing claimed skill
#     'experience_mismatch_penalty': -0.2 # Penalty for experience discrepancy
# }

# # Replace the assess_confidence_level function with this enhanced version:
# def assess_confidence_level(evaluation_scores, answers, conversation):
#     """
#     Enhanced confidence assessment that considers answer quality and skips
#     Returns: (confidence_level, decision, need_more_questions, focus_areas, reasoning)
#     """
#     if not evaluation_scores:
#         return 0.0, "Need More Information", True, [], "Initial assessment needed"
    
#     # Count skips and analyze answers
#     skipped_count = sum(1 for ans in answers.values() if ans == "Skipped")
#     scores_list = list(evaluation_scores.values())
#     poor_answers = sum(1 for score in scores_list if score < 0.6)
#     perfect_answers = sum(1 for score in scores_list if score >= 0.9)
    
#     # Calculate base confidence
#     avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
#     base_confidence = avg_score * 0.7  # Base confidence from average score

#     # Include resume consistency in confidence calculation
#     resume_consistency_score = st.session_state.get('resume_consistency_score', 1.0)
#     base_confidence *= resume_consistency_score
    
#     # Apply penalties and bonuses
#     confidence_adjustments = 0.0
#     for score in scores_list:
#         if score >= 0.9:
#             confidence_adjustments += CONFIDENCE_THRESHOLDS['perfect_answer']
#         elif score >= 0.7:
#             confidence_adjustments += CONFIDENCE_THRESHOLDS['good_answer']
#         elif score < 0.6:
#             confidence_adjustments += CONFIDENCE_THRESHOLDS['poor_answer']
    
#     # Apply skip penalties
#     confidence_adjustments += skipped_count * CONFIDENCE_THRESHOLDS['skip_penalty']
    
#     # Calculate final confidence
#     final_confidence = min(max(base_confidence + confidence_adjustments, 
#                              CONFIDENCE_THRESHOLDS['min_confidence']),
#                          CONFIDENCE_THRESHOLDS['max_confidence'])
    
#     # Determine decision and whether to continue
#     need_more_questions = True
#     focus_areas = []
#     decision = "Need More Information"
#     reasoning = ""
    
#     # Early termination conditions
#     if skipped_count >= CONFIDENCE_THRESHOLDS['skip_threshold']:
#         decision = "No Hire"
#         need_more_questions = False
#         reasoning = "Too many skipped questions indicates lack of knowledge or preparation"
#     elif poor_answers >= CONFIDENCE_THRESHOLDS['poor_answer_threshold']:
#         decision = "No Hire"
#         need_more_questions = False
#         reasoning = "Multiple poor answers indicate insufficient technical knowledge"
#     elif perfect_answers >= 3 and avg_score >= 0.85:
#         decision = "Strong Hire"
#         need_more_questions = False
#         reasoning = "Consistent excellent performance across multiple questions"
#     elif final_confidence >= CONFIDENCE_THRESHOLDS['completion_threshold']:
#         need_more_questions = False
#         decision = "Strong Hire" if avg_score >= 0.85 else "Hire" if avg_score >= 0.75 else "Lean Hire"
#         reasoning = f"Sufficient confidence reached with average score of {avg_score*100:.1f}%"
#     else:
#         # Determine focus areas for next questions
#         focus_areas = determine_focus_areas(evaluation_scores, answers)

#     # Add resume findings to reasoning if significant discrepancies found
#     if resume_consistency_score < 0.8:
#         findings = st.session_state.get('resume_findings', [])
#         if findings:
#             reasoning += "\nNote: Some inconsistencies found between resume and provided information."
        
#     return final_confidence, decision, need_more_questions, focus_areas, reasoning

# def determine_focus_areas(evaluation_scores, answers):
#     """Helper function to determine areas needing more investigation"""
#     focus_areas = []
    
#     # Analyze answer patterns
#     low_score_topics = []
#     for question, score in evaluation_scores.items():
#         if score < 0.7:
#             # Extract key technical terms from the question
#             question_lower = question.lower()
#             technical_terms = extract_technical_terms(question_lower)
#             low_score_topics.extend(technical_terms)
    
#     # Identify most common weak areas
#     if low_score_topics:
#         from collections import Counter
#         topic_counts = Counter(low_score_topics)
#         focus_areas = [topic for topic, count in topic_counts.most_common(3)]
    
#     # Add general areas if specific topics aren't clear
#     if not focus_areas:
#         focus_areas = ["problem-solving", "technical depth", "implementation details"]
    
#     return focus_areas

# def extract_technical_terms(text):
#     """Helper function to extract technical terms from text"""
#     common_tech_terms = {
#         'algorithm', 'data structure', 'optimization', 'complexity',
#         'database', 'architecture', 'design pattern', 'api',
#         'performance', 'scalability', 'security', 'testing',
#         'debugging', 'implementation', 'framework', 'library'
#     }
    
#     return [term for term in common_tech_terms if term in text]


# def generate_focused_question(tech_stack, focus_areas, previous_questions, conversation):
#     """Generate a new question based on focus areas and previous questions"""
#     focus_areas_str = ", ".join(focus_areas) if focus_areas else "general technical knowledge"
    
#     prompt = f"""
#     Based on the candidate's previous responses, generate ONE focused technical question.
#     Tech Stack: {tech_stack}
#     Focus Areas Needed: {focus_areas_str}
    
#     Previous Questions Asked:
#     {previous_questions}
    
#     Generate a NEW question that:
#     1. Probes deeper into the identified focus areas
#     2. Is different from previous questions
#     3. Helps assess technical depth and problem-solving
    
#     Return ONLY the question text, no additional formatting or commentary.
#     """
    
#     try:
#         new_question = conversation.predict(input=prompt).strip()
#         # Verify it's not too similar to previous questions
#         if any(similar_questions(new_question, prev_q) for prev_q in previous_questions):
#             # Try one more time with explicit differentiation
#             prompt += "\nIMPORTANT: Question must be substantially different from previous questions!"
#             new_question = conversation.predict(input=prompt).strip()
        
#         return new_question
#     except Exception as e:
#         return f"Error generating question: {str(e)}"

# def similar_questions(q1, q2):
#     """Basic similarity check between questions"""
#     q1_words = set(q1.lower().split())
#     q2_words = set(q2.lower().split())
#     common_words = q1_words.intersection(q2_words)
#     similarity = len(common_words) / max(len(q1_words), len(q2_words))
#     return similarity > 0.7


# def get_display_metrics(is_admin_view=False):
#     """
#     Determines which metrics to display based on user type.
#     Returns a dictionary of metrics that should be visible.
#     """
#     metrics = {
#         "questions_asked": st.session_state.questions_asked,
#     }
    
#     if is_admin_view and 'confidence_level' in st.session_state:
#         metrics["confidence_level"] = st.session_state.confidence_level
#         metrics["current_decision"] = st.session_state.get('current_decision', 'In Progress')
    
#     return metrics


# def update_assessment_progress(container, is_admin_view=False):
#     """
#     Updates the assessment progress display based on user type.
#     Uses a separate container to manage visibility.
#     """
#     metrics = get_display_metrics(is_admin_view)
    
#     with container:
#         container.empty()  # Clear previous content
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.metric("Questions Asked", metrics["questions_asked"])
        
#         if is_admin_view and "confidence_level" in metrics:
#             with col2:
#                 st.metric("Internal Confidence", f"{metrics['confidence_level']*100:.1f}%")

# def create_progress_container():
#     """Creates a container for progress metrics that can be updated dynamically"""
#     return st.empty()


def main():

    initialize_session_state()
    # Add at the beginning of main()
    try:
        llm = LLMManager.get_llm('conversation')
    except Exception as e:
        st.error(f"Failed to initialize AI components: {str(e)}")
        st.stop()
    # Determine current stage for sidebar
    if not st.session_state.get('candidate_info'):
        current_stage = 'info'
    elif not st.session_state.get('assessment_completed'):
        current_stage = 'assessment'
    else:
        current_stage = 'report'
    
    # Get resume analysis results if available
    resume_analysis = {
        'consistency_score': st.session_state.get('resume_consistency_score', 0),
        'strengths': st.session_state.get('resume_findings', []),
    } if 'resume_consistency_score' in st.session_state else None
    
    # Render sidebar with current stage and analysis
    render_sidebar(current_stage, resume_analysis)
    
    st.title('TalentScout Hiring Assistant 💼')

    # Initialize LangChain components with automated persona selection
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name='llama-3.3-70b-versatile',
            temperature=0.7,
            max_tokens=2000
        )

        memory = ConversationBufferWindowMemory(
            k=CONVERSATION_MEMORY_LENGTH,
            return_messages=True
        )

        # Automatically select persona based on candidate's experience and position
        def determine_optimal_persona(candidate_info):
            if not candidate_info:
                return 'Default'
            
            years_exp = candidate_info.get('Years of Experience', 0)
            position = candidate_info.get('Desired Position', '').lower()
            tech_stack = candidate_info.get('Tech Stack', [])
            
            # Senior/Architect positions or 8+ years experience get Expert persona
            if years_exp >= 8 or any(role in position for role in ['senior', 'lead', 'architect', 'principal']):
                return 'Expert'
            
            # Research/Innovation roles or complex tech stack get Analytical persona
            if any(role in position for role in ['research', 'data', 'ml', 'ai']) or \
               any(tech in ['machine learning', 'ai', 'data science'] for tech in tech_stack):
                return 'Analytical'
            
            # Design/UI/Creative roles get Creative persona
            if any(role in position for role in ['design', 'ui', 'ux', 'frontend', 'creative']):
                return 'Creative'
            
            # Default for other cases
            return 'Default'

        # Set the persona based on candidate info
        selected_persona = determine_optimal_persona(st.session_state.get('candidate_info', {}))
        st.session_state.selected_persona = selected_persona
        
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=get_persona_prompt(selected_persona)
        )
    except Exception as e:
        st.error(f"Error initializing AI components: {str(e)}")
        st.stop()

    # Phase 1: Initial Information Gathering
    if not st.session_state.candidate_info:
        st.header('📋 Candidate Information')
        with st.form('info_form'):
            full_name = st.text_input('Full Name*', value=st.session_state.get('full_name', ''))
            email = st.text_input('Email Address*', value=st.session_state.get('email', ''))
            phone = st.text_input('Phone Number*', value=st.session_state.get('phone', ''))
            years_exp = st.number_input('Years of Experience', min_value=0, max_value=50, step=1, value=st.session_state.get('years_exp', 0))
            desired_position = st.text_input('Desired Position(s)*', value=st.session_state.get('desired_position', ''))
            location = st.text_input('Current Location*', value=st.session_state.get('location', ''))
            tech_stack = st.text_area('Tech Stack (e.g., Python, Django, JavaScript)*', value=st.session_state.get('tech_stack', ''))
            uploaded_file = st.file_uploader(
                "Upload Resume (PDF or DOCX)*", 
                type=['pdf', 'docx'],
                help="Please upload your resume in PDF or DOCX format"
            )
            submitted = st.form_submit_button('Submit Information 📤')

            if submitted:
                # Validate all required fields
                validation_errors = []

                if not full_name.strip():
                    validation_errors.append("Full Name is required")
                if not email.strip() or not validate_email(email):
                    validation_errors.append("Valid Email Address is required")
                if not phone.strip() or not validate_phone(phone):
                    validation_errors.append("Valid Phone Number is required")
                if not desired_position.strip():
                    validation_errors.append("Desired Position is required")
                if not location.strip():
                    validation_errors.append("Location is required")
                if not tech_stack.strip() or not validate_tech_stack(tech_stack):
                    validation_errors.append("At least one Technology in Tech Stack is required")
                if not uploaded_file:
                    validation_errors.append("Resume is required")
                if validation_errors:
                    st.error("Please fix the following errors:\n" + "\n".join(validation_errors))
                else:
                    resume_text = extract_text_from_resume(uploaded_file)
                    if resume_text:
                        consistency_score, findings = analyze_resume_consistency(
                            resume_text,
                            {
                                "Full Name": full_name,
                                "Tech Stack": [tech.strip() for tech in tech_stack.split(',') if tech.strip()],
                                "Years of Experience": years_exp,
                                "Desired Position": desired_position
                            }
                        )
                    st.session_state.resume_consistency_score = consistency_score
                    st.session_state.resume_findings = findings    
                    st.session_state.candidate_info = {
                        "Full Name": full_name,
                        "Email": email,
                        "Phone": phone,
                        "Years of Experience": years_exp,
                        "Desired Position": desired_position,
                        "Location": location,
                        "Tech Stack": [tech.strip() for tech in tech_stack.split(',') if tech.strip()]
                    }
                    st.success('Information submitted successfully! 🎉')
                    st.rerun()

        st.markdown("*Required fields are marked with an asterisk (\*)")

    # Phase 2: Technical Assessment
    elif not st.session_state.assessment_completed:
        st.header('🛠️ Technical Assessment')
        
        # Create a container for progress metrics
        progress_container = create_progress_container()
        
        # Initialize assessment state if needed
        if 'assessment_state' not in st.session_state:
            st.session_state.assessment_state = {
                'internal_confidence': 0.0,
                'admin_view': False  # Could be set based on authentication
            }
        
        # Add early completion option with hidden confidence
        if st.session_state.questions_asked > 0:
            if st.button('Complete Assessment Early 🎯', help='Finish the assessment now with current results'):
                confidence, decision, _, _, reasoning = assess_confidence_level(
                    st.session_state.evaluation_scores,
                    st.session_state.answers,
                    conversation
                )
                
                # Update internal state without displaying
                st.session_state.confidence_level = confidence
                st.session_state.current_decision = decision
                st.session_state.assessment_completed = True
                st.session_state.final_reasoning = reasoning
                
                # Show completion confirmation without revealing confidence
                st.success("Assessment completed successfully!")
                st.rerun()
        
        # Update progress display
        update_assessment_progress(progress_container, st.session_state.assessment_state['admin_view'])
        
        # Generate or display current question
        if not st.session_state.current_question:
            if st.session_state.questions_asked == 0:
                # Initial questions generation
                tech_stack_str = ', '.join(st.session_state.candidate_info["Tech Stack"])
                technical_questions = generate_technical_questions(tech_stack_str, conversation)
                if not technical_questions:
                    st.error("No technical questions generated. Please check the tech stack and try again.")
                    st.stop()
                st.session_state.technical_questions = technical_questions
                st.session_state.current_question_index = 0
                st.session_state.current_question = technical_questions[0]
            else:
                # Generate focused question based on confidence assessment
                confidence, decision, need_more, focus_areas, reasoning = assess_confidence_level(
                    st.session_state.evaluation_scores,
                    st.session_state.answers,
                    conversation
                )
                
                # Update internal state without displaying
                st.session_state.confidence_level = confidence
                st.session_state.current_decision = decision
                
                # Check for assessment completion
                if not need_more or st.session_state.questions_asked >= 15:
                    st.session_state.assessment_completed = True
                    st.session_state.final_reasoning = reasoning
                    st.success("Assessment completed successfully!")
                    st.rerun()
                
                # Generate next question
                previous_questions = list(st.session_state.answers.keys())
                new_question = generate_focused_question(
                    st.session_state.candidate_info["Tech Stack"],
                    focus_areas,
                    previous_questions,
                    conversation
                )
                st.session_state.current_question = new_question
        
        # Display current question and handle response
        st.subheader(f'Question {st.session_state.questions_asked + 1}')
        st.write(st.session_state.current_question)
        
        answer = st.text_area('Your Answer 📝', 
                            value=st.session_state.get('current_answer', ''),
                            height=150, 
                            key=f"answer_{st.session_state.questions_asked}")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button('Submit Answer ✅'):
                if not answer.strip():
                    st.warning('Please provide an answer before submitting.')
                else:
                    question = st.session_state.current_question
                    st.session_state.answers[question] = answer
                    
                    # Evaluate answer
                    score, feedback = evaluate_answer_with_llm(
                        question,
                        answer,
                        st.session_state.candidate_info["Tech Stack"]
                    )
                    st.session_state.evaluation_scores[question] = score
                    
                    # Show feedback
                    if score >= 0.8:
                        st.success("Excellent answer! 🌟")
                    elif score >= 0.6:
                        st.info("Good answer with room for improvement.")
                    else:
                        st.warning("The answer needs more detail and technical depth.")
                    
                    with st.expander("View Detailed Feedback"):
                        for point in feedback:
                            st.write(f"• {point}")
                    
                    # Update assessment state
                    st.session_state.questions_asked += 1
                    st.session_state.current_question = None
                    st.session_state.current_answer = ''
                    st.rerun()
        
        with col2:
            if st.button('Skip Question ⏭️'):
                question = st.session_state.current_question
                st.session_state.answers[question] = "Skipped"
                st.session_state.evaluation_scores[question] = 0.0
                st.session_state.questions_asked += 1
                st.session_state.current_question = None
                st.session_state.current_answer = ''
                
                # Count skipped questions
                skipped_count = sum(1 for ans in st.session_state.answers.values() if ans == "Skipped")
                if skipped_count >= CONFIDENCE_THRESHOLDS['skip_threshold']:
                    st.warning("Too many questions skipped. Completing assessment.")
                    st.session_state.current_decision = "No Hire"
                    st.session_state.assessment_completed = True
                st.rerun()
    # Phase 3: Final Report and Recommendation
    else:
        st.header('📈 Assessment Report')

        # Calculate overall metrics
        if st.session_state.evaluation_scores:
            total_score = sum(st.session_state.evaluation_scores.values())
            total_questions = len(st.session_state.technical_questions)
            avg_score = total_score / total_questions if total_questions > 0 else 0
        else:
            avg_score = 0

        # Generate recommendation based on comprehensive evaluation
        if len(st.session_state.evaluation_scores) > 0:
            recommendation = generate_final_recommendation_with_llm(
                st.session_state.candidate_info,
                st.session_state.answers,
                st.session_state.evaluation_scores
            )
        else:
            recommendation = "No questions evaluated yet."

        st.session_state.recommendation = recommendation
        # Display candidate information
        st.subheader('👤 Candidate Information')
        for key, value in st.session_state.candidate_info.items():
            if isinstance(value, list):
                st.write(f"**{key}:** {', '.join(value)}")
            else:
                st.write(f"**{key}:** {value}")

        # Display technical assessment results
        st.subheader('🎯 Technical Assessment Results')

        # Create metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Average Score",
                value=f"{avg_score*100:.1f}%",
                delta=f"{(avg_score-0.7)*100:.1f}%" if avg_score > 0.7 else f"{(avg_score-0.7)*100:.1f}%"
            )
        with col2:
            st.metric(
                label="Questions Completed",
                value=f"{len(st.session_state.answers)}/{len(st.session_state.technical_questions)}"
            )
        with col3:
            highest_score = max(st.session_state.evaluation_scores.values(), default=0)
            st.metric(
                label="Highest Score",
                value=f"{highest_score*100:.1f}%"
            )

        # Detailed question analysis
        st.subheader('📝 Detailed Analysis')
        for idx, (question, answer) in enumerate(st.session_state.answers.items(), 1):
            with st.expander(f"Question {idx}"):
                st.write("**Question:**")
                st.write(question)
                st.write("**Answer:**")
                st.write(answer)
                score = st.session_state.evaluation_scores.get(question, 0)
                st.progress(score)
                st.write(f"Score: {score*100:.1f}%")

        # Display recommendation
        st.subheader('🎯 Recommendation')
        st.write(recommendation)

        # Generate and offer report download
        report = generate_report(
            st.session_state.candidate_info,
            st.session_state.answers,
            st.session_state.evaluation_scores,
            st.session_state.recommendation
        )

        # Add download buttons for different formats
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label='Download JSON Report 📥',
                data=report,
                file_name=f"{st.session_state.candidate_info['Full Name'].replace(' ', '_')}_Assessment_Report.json",
                mime='application/json'
            )

        with col2:
            # Create PDF-friendly format
            pdf_report = f"""
            TalentScout Assessment Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Candidate Information:
            {json.dumps(st.session_state.candidate_info, indent=2)}

            Technical Assessment Results:
            Average Score: {avg_score*100:.1f}%
            Questions Completed: {len(st.session_state.answers)}/{len(st.session_state.technical_questions)}

            Recommendation:
            {recommendation}
            """

            st.download_button(
                label='Download Text Report 📄',
                data=pdf_report,
                file_name=f"{st.session_state.candidate_info['Full Name'].replace(' ', '_')}_Assessment_Report.txt",
                mime='text/plain'
            )

    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")