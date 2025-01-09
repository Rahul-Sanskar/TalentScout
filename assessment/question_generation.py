import streamlit as st
from langchain.chains import ConversationChain
import re

# Function to generate technical questions
def generate_technical_questions(tech_stack, conversation):
    prompt = f"""
    Based on the tech stack: {tech_stack}, generate 5 questions with increasing difficulty:

    1. Start with a basic concept/definition question (easy, short answer)
    2. Progress to fundamentals application (moderate, brief explanation)
    3. Add a practical scenario (moderate, focused solution)
    4. Include problem-solving (challenging but specific)
    5. End with advanced concepts (if needed based on previous answers)

    Rules:
    - First 2 questions should be answerable in 1-2 sentences
    - Questions 3-4 should need 3-4 sentences max
    - Keep questions focused and specific
    - Avoid asking for code implementations
    - Use this format: "Question N: [The question text]"

    Example progression:
    Question 1: What is [basic concept] in {tech_stack}?
    Question 2: How does [fundamental feature] work in {tech_stack}?
    Question 3: In a simple web app, how would you handle [specific scenario]?
    Question 4: What approach would you take to solve [specific problem]?
    Question 5: Explain the trade-offs between [advanced concept A] and [advanced concept B].

    Generate questions that are concise and clear.
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


def generate_focused_question(tech_stack, focus_areas, previous_questions, conversation):
    """Generate a new question based on focus areas and previous questions"""
    focus_areas_str = ", ".join(focus_areas) if focus_areas else "general technical knowledge"
    
    prompt = f"""
    Based on the candidate's previous responses, generate ONE focused technical question.
    Tech Stack: {tech_stack}
    Focus Areas Needed: {focus_areas_str}
    
    Previous Questions Asked:
    {previous_questions}
    
    Generate a NEW question that:
    1. Probes deeper into the identified focus areas
    2. Is different from previous questions
    3. Helps assess technical depth and problem-solving
    
    Return ONLY the question text, no additional formatting or commentary.
    """
    
    try:
        new_question = conversation.predict(input=prompt).strip()
        # Verify it's not too similar to previous questions
        if any(similar_questions(new_question, prev_q) for prev_q in previous_questions):
            # Try one more time with explicit differentiation
            prompt += "\nIMPORTANT: Question must be substantially different from previous questions!"
            new_question = conversation.predict(input=prompt).strip()
        
        return new_question
    except Exception as e:
        return f"Error generating question: {str(e)}"

def similar_questions(q1, q2):
    """Basic similarity check between questions"""
    q1_words = set(q1.lower().split())
    q2_words = set(q2.lower().split())
    common_words = q1_words.intersection(q2_words)
    similarity = len(common_words) / max(len(q1_words), len(q2_words))
    return similarity > 0.7        