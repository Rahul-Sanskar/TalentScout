# Create a new file: llm_manager.py

from langchain_groq import ChatGroq
from typing import Optional, Dict, Any
import streamlit as st

class LLMManager:
    """
    Singleton class to manage LLM instances with different configurations.
    Provides centralized control over LLM creation and caching.
    """
    _instances: Dict[str, ChatGroq] = {}
    
    @classmethod
    def get_llm(cls, llm_type: str, **kwargs) -> ChatGroq:
        """
        Get or create an LLM instance based on type and configuration.
        
        Args:
            llm_type: Type of LLM configuration ('evaluation', 'conversation', 'recommendation', 'report')
            **kwargs: Optional override parameters for the LLM configuration
            
        Returns:
            ChatGroq: Configured LLM instance
        """
        # Define base configurations for different LLM types
        base_configs = {
            'evaluation': {
                'temperature': 0.4,
                'max_tokens': 4028,
                'top_p': 0.95,
                'presence_penalty': 0.6,
                'frequency_penalty': 0.3
            },
            'conversation': {
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 1.0,
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0
            },
            'recommendation': {
                'temperature': 0.5,
                'max_tokens': 4028,
                'top_p': 0.9,
                'presence_penalty': 0.4,
                'frequency_penalty': 0.4
            },
            'report': {
                'temperature': 0.3,
                'max_tokens': 4028,
                'top_p': 0.8,
                'presence_penalty': 0.2,
                'frequency_penalty': 0.2
            }
        }
        
        # Create cache key based on configuration
        config = {**base_configs.get(llm_type, {}), **kwargs}
        cache_key = f"{llm_type}_{hash(frozenset(config.items()))}"
        
        # Return cached instance if it exists
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        try:
            # Create new instance with provided configuration
            llm = ChatGroq(
                api_key=st.secrets["GROQ_API_KEY"],
                model_name='llama-3.3-70b-versatile',
                **config
            )
            cls._instances[cache_key] = llm
            return llm
            
        except Exception as e:
            raise RuntimeError(f"Failed to create LLM instance: {str(e)}")
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached LLM instances"""
        cls._instances.clear()