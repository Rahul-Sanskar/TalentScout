import streamlit as st
from utils.resume_processing import generate_motivation_message
from config.settings import CONFIDENCE_THRESHOLDS

def create_assessment_guidelines():
    """Create consistent assessment guidelines for sidebar"""
    guidelines = {
        "steps": [
            "📋 Initial Information Collection",
            "📎 Resume Analysis & Verification",
            "🔍 Technical Skills Assessment",
            "📊 Performance Evaluation",
            "📈 Final Report Generation"
        ],
        "rules": [
            "Answer each question thoroughly and precisely",
            "Take time to structure your responses clearly",
            "Focus on practical experience and examples",
            "Be honest about knowledge limitations",
            "Maintain professional communication"
        ]
    }
    return guidelines

def render_sidebar(stage, resume_analysis=None):
    """Render consistent sidebar content across all stages"""
    with st.sidebar:
        st.header('Assessment Guide 📚')
        
        # Display current stage
        stages = {
            'info': '📋 Information Collection',
            'assessment': '🔍 Technical Assessment',
            'report': '📈 Final Report'
        }
        current_stage = stages.get(stage, '')
        st.subheader(f"Current Stage: {current_stage}")
        
        # Display guidelines
        guidelines = create_assessment_guidelines()
        
        with st.expander("📝 Assessment Steps", expanded=True):
            for idx, step in enumerate(guidelines['steps'], 1):
                if stages[stage] in step:
                    st.markdown(f"**→ {idx}. {step}**")
                else:
                    st.markdown(f"{idx}. {step}")
        
        with st.expander("📋 Assessment Rules", expanded=True):
            for rule in guidelines['rules']:
                st.markdown(f"• {rule}")
        
        # Display motivation if resume has been analyzed
        if resume_analysis and 'consistency_score' in resume_analysis:
            st.markdown("---")
            st.subheader("💫 Your Assessment Journey")
            motivation = generate_motivation_message(resume_analysis)
            st.markdown(f"*{motivation}*")
        
        # Clear session button at bottom
        st.markdown("---")
        if st.button('Reset Assessment 🔄'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
