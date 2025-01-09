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