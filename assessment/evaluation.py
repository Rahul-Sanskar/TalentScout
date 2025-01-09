import json
import streamlit as st
from config.settings import CONFIDENCE_THRESHOLDS
from models.llm_manager import LLMManager
from datetime import datetime
from collections import Counter
import traceback

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


def evaluate_answer_with_llm(question, answer, tech_stack):
    """Evaluate answer using LLM with improved response handling"""
    evaluation_llm = LLMManager.get_llm('evaluation')
    
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

def generate_detailed_feedback_with_llm(answers, tech_stack):
    """Generate comprehensive feedback using LLM"""
    feedback_llm = LLMManager.get_llm('evaluation')
    
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
    
    Format your response as detailed but precise and to the point paragraphs.
    Make sure you Explore complex thoughts through introspective, analytical, and philosophical self-examination but provide simple and clear and short feedback.
    """
    
    try:
        detailed_feedback = feedback_llm.predict(prompt)
        return detailed_feedback
    except Exception as e:
        return f"Error generating detailed feedback: {str(e)}"


def generate_final_recommendation_with_llm(candidate_info, answers, scores):
    """Generate final recommendation using LLM with enhanced prompting and fallback"""
    recommendation_llm = LLMManager.get_llm('recommendation')
    
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


def assess_confidence_level(evaluation_scores, answers, conversation):
    """
    Enhanced confidence assessment that considers answer quality and skips
    Returns: (confidence_level, decision, need_more_questions, focus_areas, reasoning)
    """
    if not evaluation_scores:
        return 0.0, "Need More Information", True, [], "Initial assessment needed"
    
    # Count skips and analyze answers
    skipped_count = sum(1 for ans in answers.values() if ans == "Skipped")
    scores_list = list(evaluation_scores.values())
    poor_answers = sum(1 for score in scores_list if score < 0.6)
    perfect_answers = sum(1 for score in scores_list if score >= 0.9)
    
    # Calculate base confidence
    avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
    base_confidence = avg_score * 0.7  # Base confidence from average score

    # Include resume consistency in confidence calculation
    resume_consistency_score = st.session_state.get('resume_consistency_score', 1.0)
    base_confidence *= resume_consistency_score
    
    # Apply penalties and bonuses
    confidence_adjustments = 0.0
    for score in scores_list:
        if score >= 0.9:
            confidence_adjustments += CONFIDENCE_THRESHOLDS['perfect_answer']
        elif score >= 0.7:
            confidence_adjustments += CONFIDENCE_THRESHOLDS['good_answer']
        elif score < 0.6:
            confidence_adjustments += CONFIDENCE_THRESHOLDS['poor_answer']
    
    # Apply skip penalties
    confidence_adjustments += skipped_count * CONFIDENCE_THRESHOLDS['skip_penalty']
    
    # Calculate final confidence
    final_confidence = min(max(base_confidence + confidence_adjustments, 
                             CONFIDENCE_THRESHOLDS['min_confidence']),
                         CONFIDENCE_THRESHOLDS['max_confidence'])
    
    # Determine decision and whether to continue
    need_more_questions = True
    focus_areas = []
    decision = "Need More Information"
    reasoning = ""
    
    # Early termination conditions
    if skipped_count >= CONFIDENCE_THRESHOLDS['skip_threshold']:
        decision = "No Hire"
        need_more_questions = False
        reasoning = "Too many skipped questions indicates lack of knowledge or preparation"
    elif poor_answers >= CONFIDENCE_THRESHOLDS['poor_answer_threshold']:
        decision = "No Hire"
        need_more_questions = False
        reasoning = "Multiple poor answers indicate insufficient technical knowledge"
    elif perfect_answers >= 3 and avg_score >= 0.85:
        decision = "Strong Hire"
        need_more_questions = False
        reasoning = "Consistent excellent performance across multiple questions"
    elif final_confidence >= CONFIDENCE_THRESHOLDS['completion_threshold']:
        need_more_questions = False
        decision = "Strong Hire" if avg_score >= 0.85 else "Hire" if avg_score >= 0.75 else "Lean Hire"
        reasoning = f"Sufficient confidence reached with average score of {avg_score*100:.1f}%"
    else:
        # Determine focus areas for next questions
        focus_areas = determine_focus_areas(evaluation_scores, answers)

    # Add resume findings to reasoning if significant discrepancies found
    if resume_consistency_score < 0.8:
        findings = st.session_state.get('resume_findings', [])
        if findings:
            reasoning += "\nNote: Some inconsistencies found between resume and provided information."
        
    return final_confidence, decision, need_more_questions, focus_areas, reasoning

def determine_focus_areas(evaluation_scores, answers):
    """Helper function to determine areas needing more investigation"""
    focus_areas = []
    
    # Analyze answer patterns
    low_score_topics = []
    for question, score in evaluation_scores.items():
        if score < 0.7:
            # Extract key technical terms from the question
            question_lower = question.lower()
            technical_terms = extract_technical_terms(question_lower)
            low_score_topics.extend(technical_terms)
    
    # Identify most common weak areas
    if low_score_topics:
        from collections import Counter
        topic_counts = Counter(low_score_topics)
        focus_areas = [topic for topic, count in topic_counts.most_common(3)]
    
    # Add general areas if specific topics aren't clear
    if not focus_areas:
        focus_areas = ["problem-solving", "technical depth", "implementation details"]
    
    return focus_areas

def extract_technical_terms(text):
    """Helper function to extract technical terms from text"""
    common_tech_terms = {
        'algorithm', 'data structure', 'optimization', 'complexity',
        'database', 'architecture', 'design pattern', 'api',
        'performance', 'scalability', 'security', 'testing',
        'debugging', 'implementation', 'framework', 'library'
    }
    
    return [term for term in common_tech_terms if term in text]
