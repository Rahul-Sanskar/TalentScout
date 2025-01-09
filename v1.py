import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from datetime import datetime
import os
import json
import re

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
        'current_answer': ''  # Add this new state variable
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default


# Input validation functions
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    # Basic phone validation - can be adjusted based on your needs
    pattern = r'^\+?1?\d{9,15}$'
    return bool(re.match(pattern, phone))

def validate_tech_stack(tech_stack):
    techs = [tech.strip() for tech in tech_stack.split(',') if tech.strip()]
    return len(techs) > 0

# Define persona-based prompt templates
def get_persona_prompt(persona):
    personas = {
    'Default': ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and professional hiring assistant. 
                     Your role is to conduct preliminary technical screenings for candidates. 
                     Focus on gathering essential details, maintaining a conversational tone, 
                     and assessing both technical knowledge and problem-solving abilities. 
                     Provide constructive feedback without overwhelming the candidate."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]),
    
    'Expert': ChatPromptTemplate.from_messages([
        ("system", """You are a highly experienced technical hiring manager. 
                     Your job is to assess candidates thoroughly on:
                     - Technical accuracy
                     - Problem-solving strategies
                     - Code quality and optimization
                     - System design and scalability
                     Start with foundational questions, then dive into advanced topics 
                     and edge cases. Offer precise, actionable feedback based on the 
                     candidate's responses, highlighting strengths and improvement areas."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]),
    
    'Creative': ChatPromptTemplate.from_messages([
        ("system", """You are an engaging and innovative interviewer who evaluates 
                     candidates through real-world scenarios and practical challenges. 
                     Assess:
                     - Creative problem-solving
                     - Adaptability to unique scenarios
                     - Application of technical knowledge
                     - Clear and concise communication
                     Use situational questions and collaborative problem-solving exercises 
                     to encourage critical thinking."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]),

    'Analytical': ChatPromptTemplate.from_messages([
        ("system", """You are a data-driven and analytical evaluator. 
                     Your focus is on assessing logical reasoning and analytical skills 
                     alongside technical expertise. Start with short and specific 
                     questions, progressing to scenarios that require deeper analysis. 
                     Evaluate based on:
                     - Clarity in logic
                     - Efficiency in problem-solving
                     - Ability to break down complex problems into manageable steps."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]),
}
    return personas.get(persona, personas['Default'])

# Function to generate technical questions
def generate_technical_questions(tech_stack, conversation):
    # prompt = f"""
    # Based on the tech stack: {tech_stack}, generate 5 diverse technical interview questions following this progression:

    # 1. Start with a fundamental concept question (one-liner or MCQ)
    # 2. Move to a practical implementation question
    # 3. Present a problem-solving scenario
    # 4. Ask about system design or architecture
    # 5. Explore best practices and optimization

    # Format rules:
    # - Begin each with "Question N:"
    # - Mix question types (MCQ, coding, theoretical, scenario-based)
    # - Keep scenarios concise but realistic
    # - Focus on core concepts first, then advanced topics
    # - Include both practical and theoretical aspects

    # Example structure:
    # Question 1: [MCQ] Which of these {tech_stack} concepts...?
    # A) Option 1  B) Option 2  C) Option 3  D) Option 4

    # Question 2: In a web application using {tech_stack}, implement...

    # Question 3: Given a production scenario where...

    # Keep responses strictly in this format without additional commentary.
    # Generate questions that evaluate both knowledge depth and problem-solving ability.
    # """
    prompt = f"""
    For the following tech stack: {tech_stack}
    Generate 5 technical interview questions. For each question:
    1. Start with "Question N:" where N is the question number
    2. Provide a brief scenario
    3. Ask a specific technical question
    
    Keep the format simple and consistent. Do not include additional headers or sections.
    Example format:
    Question 1: In a system using {tech_stack}, how would you implement...?
    Question 2: Given a scenario where...?
    """
   
    try:
        response = conversation.predict(input=prompt)
        
        questions = []
        for line in response.splitlines():
            if line.strip().startswith('Question'):
                questions.append(line.strip())
        
        return questions[:5]  # Ensure we only return 5 questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return ["Error generating questions. Please try again."]







def create_evaluation_llm():
    """Create a separate LLM instance for evaluation"""
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name='llama-3.3-70b-versatile',
        temperature=0.5,  # Lower temperature for more consistent evaluation
        max_tokens=4028
    )

def evaluate_answer_with_llm(question, answer, tech_stack):
    """Evaluate answer using LLM with improved response handling"""
    evaluation_llm = create_evaluation_llm()
    
    prompt = f"""You are an expert technical interviewer evaluating a candidate's response. You must return your evaluation in the exact JSON format specified below.

Question: {question}
Candidate's Answer: {answer}
Relevant Technologies: {', '.join(tech_stack)}

Evaluate the answer across these dimensions:
1. Technical Accuracy
2. Completeness
3. Clarity
4. Best Practices

Your response must be in this exact JSON format with no additional text before or after:
{{
    "technical_accuracy": {{
        "score": <number between 0-100>,
        "feedback": "<specific feedback>"
    }},
    "completeness": {{
        "score": <number between 0-100>,
        "feedback": "<specific feedback>"
    }},
    "clarity": {{
        "score": <number between 0-100>,
        "feedback": "<specific feedback>"
    }},
    "best_practices": {{
        "score": <number between 0-100>,
        "feedback": "<specific feedback>"
    }},
    "overall_feedback": "<summarizing feedback>"
}}

Remember: Return only valid JSON, no other text."""
    
    try:
        # Get LLM response
        response = evaluation_llm.predict(prompt)
        
        # Clean the response - remove any potential markdown formatting or extra text
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        # Try to parse the JSON
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a default structured response
            st.warning("AI response formatting issue. Using simplified evaluation.")
            return fallback_evaluation(answer)
        
        # Calculate normalized scores (0-1 range)
        scores = {
            'technical_accuracy': float(evaluation['technical_accuracy']['score']) / 100,
            'completeness': float(evaluation['completeness']['score']) / 100,
            'clarity': float(evaluation['clarity']['score']) / 100,
            'best_practices': float(evaluation['best_practices']['score']) / 100
        }
        
        # Collect feedback
        feedback = [
            f"Technical Accuracy: {evaluation['technical_accuracy']['feedback']}",
            f"Completeness: {evaluation['completeness']['feedback']}",
            f"Clarity: {evaluation['clarity']['feedback']}",
            f"Best Practices: {evaluation['best_practices']['feedback']}",
            f"\nOverall: {evaluation['overall_feedback']}"
        ]
        
        # Calculate final score as weighted average
        weights = {
            'technical_accuracy': 0.4,
            'completeness': 0.2,
            'clarity': 0.2,
            'best_practices': 0.2
        }
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        return final_score, feedback
        
    except Exception as e:
        st.warning(f"Using fallback evaluation due to: {str(e)}")
        return fallback_evaluation(answer)

def fallback_evaluation(answer):
    """Provide a basic evaluation when LLM evaluation fails"""
    # Basic scoring based on answer length and complexity
    length_score = min(len(answer.split()) / 100, 1.0)  # Normalize based on word count
    
    # Basic complexity score based on technical term usage
    technical_terms = ['function', 'class', 'method', 'algorithm', 'complexity', 'performance', 'optimization']
    tech_term_count = sum(1 for term in technical_terms if term.lower() in answer.lower())
    complexity_score = min(tech_term_count / 5, 1.0)
    
    # Calculate final score
    final_score = (length_score + complexity_score) / 2
    
    # Generate basic feedback
    feedback = [
        f"Answer length: {'Good' if length_score > 0.7 else 'Could be more detailed'}",
        f"Technical depth: {'Good' if complexity_score > 0.7 else 'Could include more technical details'}",
        "\nOverall: The answer has been evaluated using basic metrics. Please try submitting again for a more detailed AI evaluation."
    ]
    
    return final_score, feedback

def create_evaluation_llm():
    """Create a separate LLM instance for evaluation with optimized parameters"""
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name='llama-3.3-70b-versatile',
        temperature=0.4,  # Slightly higher for more detailed responses
        max_tokens=4028,  # Increased for longer responses
        top_p=0.95,
        presence_penalty=0.6,  # Encourage more diverse responses
        frequency_penalty=0.3  # Reduce repetition
    )  


def generate_detailed_feedback_with_llm(answers, tech_stack):
    """Generate comprehensive feedback using LLM"""
    feedback_llm = create_evaluation_llm()
    
    answers_summary = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items()])
    
    prompt = f"""
    Review these technical interview answers:
    {answers_summary}
    
    Technologies: {', '.join(tech_stack)}
    
    Provide a comprehensive evaluation including:
    1. Key strengths
    2. Areas for improvement
    3. Technical proficiency level
    4. Specific recommendations
    
    Format your response as detailed paragraphs.
    """
    
    try:
        detailed_feedback = feedback_llm.predict(prompt)
        return detailed_feedback
    except Exception as e:
        return f"Error generating detailed feedback: {str(e)}"

def generate_final_recommendation_with_llm(candidate_info, answers, scores):
    """Generate final recommendation using LLM with enhanced prompting and fallback"""
    recommendation_llm = create_evaluation_llm()
    
    # Calculate key metrics for context
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    strengths = [q for q, s in scores.items() if s >= 0.7]
    areas_for_improvement = [q for q, s in scores.items() if s < 0.7]
    
    prompt = f"""As an expert technical interviewer, provide a detailed hiring recommendation. You MUST follow this specific structure in your response.

CANDIDATE PROFILE:
{json.dumps(candidate_info, indent=2)}

ASSESSMENT METRICS:
- Average Score: {avg_score * 100:.1f}%
- Questions Answered: {len(answers)}
- Strong Areas: {len(strengths)} questions
- Areas Needing Improvement: {len(areas_for_improvement)} questions

YOUR TASK:
Provide a comprehensive recommendation following EXACTLY this format:

1. RECOMMENDATION: [Must be one of: "Strong Hire", "Hire", "Hold - Need More Information", "No Hire"]

2. JUSTIFICATION:
- Technical Skills Assessment
- Problem Solving Abilities
- Communication Quality
- Overall Fit for Role

3. KEY STRENGTHS:
- [List at least 3 specific strengths]

4. AREAS FOR IMPROVEMENT:
- [List at least 2 specific areas]

5. SUGGESTED NEXT STEPS:
- [Provide at least 3 specific actionable steps]

You MUST fill out all sections in detail. Keep your response professional and constructive, even for "No Hire" recommendations.
"""

    try:
        recommendation = recommendation_llm.predict(prompt)
        
        # Verify if the response has all required sections
        required_sections = [
            "RECOMMENDATION:", 
            "JUSTIFICATION:", 
            "KEY STRENGTHS:", 
            "AREAS FOR IMPROVEMENT:", 
            "SUGGESTED NEXT STEPS:"
        ]
        
        if not all(section in recommendation for section in required_sections):
            # If missing sections, try one more time with a more forceful prompt
            return generate_fallback_recommendation(candidate_info, answers, scores)
            
        return recommendation
        
    except Exception as e:
        return generate_fallback_recommendation(candidate_info, answers, scores)


def generate_fallback_recommendation(candidate_info, answers, scores):
    """Generate a structured recommendation when the primary method fails"""
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    
    # Define recommendation based on average score
    if avg_score >= 0.8:
        hire_status = "Strong Hire"
        confidence = "high"
    elif avg_score >= 0.7:
        hire_status = "Hire"
        confidence = "moderate"
    elif avg_score >= 0.5:
        hire_status = "Hold - Need More Information"
        confidence = "low"
    else:
        hire_status = "No Hire"
        confidence = "moderate"

    # Generate structured recommendation
    recommendation = f"""
1. RECOMMENDATION: {hire_status}

2. JUSTIFICATION:
- Technical Skills Assessment: Candidate demonstrated {'strong' if avg_score >= 0.7 else 'moderate' if avg_score >= 0.5 else 'insufficient'} technical knowledge
- Problem Solving Abilities: {'Effectively' if avg_score >= 0.7 else 'Adequately' if avg_score >= 0.5 else 'Insufficiently'} solved presented challenges
- Communication Quality: Responses were {'clear and well-structured' if avg_score >= 0.7 else 'adequate' if avg_score >= 0.5 else 'needing improvement'}
- Overall Fit for Role: {'Strong' if avg_score >= 0.7 else 'Potential' if avg_score >= 0.5 else 'Limited'} alignment with position requirements

3. KEY STRENGTHS:
- {'Demonstrated technical knowledge in ' + ', '.join(candidate_info.get('Tech Stack', ['relevant areas'])[:3])}
- {'Strong problem-solving approach' if avg_score >= 0.7 else 'Basic understanding of concepts'}
- {'Clear communication skills' if avg_score >= 0.6 else 'Willingness to engage with technical questions'}

4. AREAS FOR IMPROVEMENT:
- {'Advanced concepts in ' + ', '.join(candidate_info.get('Tech Stack', ['relevant areas'])[:2])}
- {'Detailed problem analysis' if avg_score < 0.8 else 'Edge case handling'}
- {'Technical communication clarity' if avg_score < 0.7 else 'Advanced scenario handling'}

5. SUGGESTED NEXT STEPS:
- {'Schedule final round interview' if hire_status == 'Strong Hire' else 'Conduct additional technical assessment' if hire_status == 'Hold - Need More Information' else 'Consider for different role/level'}
- {'Prepare system design discussion' if avg_score >= 0.7 else 'Review fundamental concepts'}
- {'Discuss team fit and project experience' if avg_score >= 0.6 else 'Gain more practical experience'}
- {'Evaluate architectural knowledge' if avg_score >= 0.8 else 'Focus on core competency development'}

Note: This recommendation is based on quantitative assessment scores and candidate profile analysis.
"""
    
    return recommendation


# Modified generate_report function
def generate_report(candidate_info, answers, evaluation_scores, recommendation):
    report_llm = create_evaluation_llm()
    
    prompt = f"""
    Generate a comprehensive assessment report for:
    
    Candidate: {json.dumps(candidate_info, indent=2)}
    Evaluation Scores: {json.dumps(evaluation_scores, indent=2)}
    Recommendation: {recommendation}
    
    Include:
    1. Executive summary
    2. Technical evaluation
    3. Key observations
    4. Next steps
    
    Format the response as a detailed JSON report.
    """
    
    try:
        report_content = report_llm.predict(prompt)
        report_json = json.loads(report_content)
        
        # Add metadata
        report_json["Report Generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_json["Candidate Information"] = candidate_info
        report_json["Technical Assessment"] = {
            question: {
                "Answer": answer,
                "Score": f"{evaluation_scores.get(question, 0) * 100:.1f}%"
            }
            for question, answer in answers.items()
        }
        
        return json.dumps(report_json, indent=4)
    except Exception as e:
        return json.dumps({
            "error": f"Error generating report: {str(e)}",
            "basic_info": {
                "candidate": candidate_info,
                "scores": evaluation_scores,
                "recommendation": recommendation
            }
        }, indent=4)



# Main application logic
def main():
    initialize_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header('Settings ⚙️')

        persona = st.selectbox(
            'Select AI Persona:',
            ['Default', 'Expert', 'Creative', 'Analytical'],
            index=0
        )
        st.session_state.selected_persona = persona

        memory_length = st.slider(
            'Conversation Memory Length:',
            min_value=1,
            max_value=10,
            value=5,
            help='Number of previous messages to remember'
        )
        st.session_state.memory_length = memory_length

        if st.button('Clear Session 🗑️'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.title('TalentScout Hiring Assistant 💼')

    # Initialize LangChain components
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name='llama-3.3-70b-versatile',
            temperature=0.7,
            max_tokens=2000
        )

        memory = ConversationBufferWindowMemory(
            k=st.session_state.memory_length,
            return_messages=True
        )

        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=get_persona_prompt(st.session_state.selected_persona)
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

                if validation_errors:
                    st.error("Please fix the following errors:\n" + "\n".join(validation_errors))
                else:
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

        if not st.session_state.current_question:
            try:
                tech_stack_str = ', '.join(st.session_state.candidate_info["Tech Stack"])
                technical_questions = generate_technical_questions(tech_stack_str, conversation)
                if not technical_questions:
                    st.error("No technical questions generated. Please check the tech stack and try again.")
                    st.stop()
                st.session_state.technical_questions = technical_questions
                st.session_state.current_question_index = 0
                st.session_state.current_question = technical_questions[0]
                st.session_state.current_answer = ''  # Initialize empty answer
            except Exception as e:
                st.error(f"Error generating technical questions: {str(e)}")
                st.stop()

        # Display progress
        progress = st.progress(st.session_state.current_question_index / len(st.session_state.technical_questions))
        st.subheader(f'Question {st.session_state.current_question_index + 1} of {len(st.session_state.technical_questions)}')
        st.write(st.session_state.current_question)

        # Answer input with key based on question index
        answer = st.text_area('Your Answer 📝', 
                            value=st.session_state.current_answer,
                            height=150, 
                            key=f"answer_{st.session_state.current_question_index}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button('Submit Answer ✅'):
                if not answer.strip():
                    st.warning('Please provide an answer before submitting.')
                else:
                    question = st.session_state.current_question
                    st.session_state.answers[question] = answer

                    # Evaluate answer using LLM
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

                    # Show specific feedback points
                    with st.expander("View Detailed Feedback"):
                        for point in feedback:
                            st.write(f"• {point}")

                    # Move to next question or complete assessment
                    st.session_state.current_question_index += 1
                    st.session_state.current_answer = ''  # Clear the answer
                    
                    if st.session_state.current_question_index < len(st.session_state.technical_questions):
                        st.session_state.current_question = st.session_state.technical_questions[
                            st.session_state.current_question_index
                        ]
                        st.rerun()
                    else:
                        st.session_state.assessment_completed = True
                        st.success("Assessment completed! 🎉")
                        st.rerun()

        with col2:
            if st.button('Skip Question ⏭️'):
                question = st.session_state.current_question
                st.session_state.answers[question] = "Skipped"
                st.session_state.evaluation_scores[question] = 0.0
                
                st.session_state.current_question_index += 1
                st.session_state.current_answer = ''  # Clear the answer
                
                if st.session_state.current_question_index < len(st.session_state.technical_questions):
                    st.session_state.current_question = st.session_state.technical_questions[
                        st.session_state.current_question_index
                    ]
                else:
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
