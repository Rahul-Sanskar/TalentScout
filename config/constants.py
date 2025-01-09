CONFIDENCE_THRESHOLDS = {
    'perfect_answer': 0.15,  # Increase for perfect answers
    'good_answer': 0.08,     # Increase for good answers
    'poor_answer': -0.12,    # Decrease for poor answers
    'skip_penalty': -0.2,    # Heavy penalty for skips
    'max_confidence': 0.95,  # Maximum possible confidence
    'min_confidence': 0.0,   # Minimum possible confidence
    'completion_threshold': 0.85,  # Confidence needed to complete
    'skip_threshold': 3,     # Maximum allowed skips
    'poor_answer_threshold': 4,  # Maximum allowed poor answers
    'resume_mismatch_penalty': -0.15,  # Penalty for inconsistencies
    'resume_match_bonus': 0.1,         # Bonus for strong matches
    'skill_mismatch_penalty': -0.08,   # Penalty for each missing claimed skill
    'experience_mismatch_penalty': -0.2 # Penalty for experience discrepancy
}

CONVERSATION_MEMORY_LENGTH = 10