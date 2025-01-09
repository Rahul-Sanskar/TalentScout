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
