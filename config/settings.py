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
            