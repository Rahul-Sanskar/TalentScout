import streamlit as st
from config.settings import CONFIDENCE_THRESHOLDS

def create_progress_container():
    """Creates a container for progress metrics that can be updated dynamically"""
    return st.empty()

def update_assessment_progress(container, is_admin_view=False):
    """
    Updates the assessment progress display based on user type.
    Uses a separate container to manage visibility.
    """
    metrics = get_display_metrics(is_admin_view)
    
    with container:
        container.empty()  # Clear previous content
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Questions Asked", metrics["questions_asked"])
        
        if is_admin_view and "confidence_level" in metrics:
            with col2:
                st.metric("Internal Confidence", f"{metrics['confidence_level']*100:.1f}%")    