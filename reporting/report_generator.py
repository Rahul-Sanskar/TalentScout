import json
from datetime import datetime
from models.llm_manager import LLMManager
import streamlit as st
def generate_report(candidate_info, answers, evaluation_scores, recommendation):
    report_llm = LLMManager.get_llm('report')
    
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