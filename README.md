# TalentScout AI - Intelligent Hiring Assistant

## Overview

TalentScout AI is an AI-driven hiring assistant designed to automate and optimize candidate screening through resume parsing, dynamic technical question generation, sentiment and multilingual analysis, and detailed performance reporting. Developed using Python and Streamlit, it integrates cutting-edge LLMs (via Groq API), sentiment scoring, and real-time candidate evaluation—all deployable on Streamlit Cloud.

This documentation outlines the architecture, features, setup, usage, deployment, and development process.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Features](#features)

   * [Functionality](#functionality)
   * [User Interface (UI)](#user-interface-ui)
   * [Chatbot Capabilities](#chatbot-capabilities)
   * [Resume Analysis](#resume-analysis)
   * [Dynamic Question Generation](#dynamic-question-generation)
   * [Sentiment Analysis](#sentiment-analysis)
   * [Confidence Scoring](#confidence-scoring)
   * [Feedback and Reporting](#feedback-and-reporting)
   * [Multilingual Support](#multilingual-support)
   * [Prompt Engineering](#prompt-engineering)
   * [State Management](#state-management)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Deployment](#deployment)
7. [Steps Taken During Development](#steps-taken-during-development)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

---

## System Architecture

TalentScout AI is built with modularity and scalability in mind:

* Streamlit interface for candidate interaction
* Resume parsing using PyPDF2 and python-docx
* Dynamic tech-question generation using Groq LLMs
* Sentiment and confidence scoring for better evaluation
* Multilingual support with auto-translation
* JSON-based reporting for feedback

---

## Features

### Functionality

* Modular Python design for ease of extension
* Resume-to-declared-info matching
* Realtime Q\&A generation from declared tech stack

### User Interface (UI)

* Intuitive Streamlit interface
* Sidebar includes instructions and motivational quotes
* Input collection fields and resume upload

### Chatbot Capabilities

* Interactive assistant with greeting, input prompts, and dynamic questions
* Session ends gracefully if skipped too often or on keyword triggers

### Resume Analysis

* Parses resumes and cross-validates with user-declared experience and skills
* Flags inconsistencies
* Computes alignment score for candidate profiling

### Dynamic Question Generation

* LLM-generated questions tailored to the candidate's skills
* Context-aware question progression
* Advanced prompt formatting and flow control

### Sentiment Analysis

* Uses TextBlob to assess tone of answers
* Classifies responses as Positive, Neutral, or Negative
* Adds emotional depth and helps interpret candidate confidence

### Confidence Scoring

* Updates after each question response
* Increases for correct, clear answers
* Decreases for poor or evasive responses
* Influences future question complexity

### Feedback and Reporting

* Full candidate evaluation stored as downloadable JSON/text
* Report includes:

  * Sentiment trends
  * Answer summaries
  * Technical performance
  * Recommendation for hiring

### Multilingual Support

* Detects user input language using `langdetect`
* Translates to English if needed before LLM processing
* Enables candidates to answer in their native language

### Prompt Engineering

* Structured prompts to Llama 3.3 70B for coherence and relevance
* Few-shot examples and step-wise reasoning included

### State Management

* Streamlit's session state maintains:

  * Candidate profile
  * Confidence score
  * Question/answer history

---

## File Structure

```bash
project/
├── main.py
├── assessment/
│   ├── question_generation.py
│   ├── evaluation.py
├── components/
│   ├── sidebar.py
│   ├── progress.py
├── config/
│   └── settings.py
├── models/
│   └── llm_manager.py
├── reporting/
│   └── report_generator.py
├── utils/
│   ├── validators.py
│   ├── resume_processing.py
├── .streamlit/
│   └── secrets.toml
```

---

## Installation

### Prerequisites

* Python 3.8+
* pip

### Steps

```bash
git clone https://github.com/yourusername/TalentScout-AI.git
cd TalentScout-AI

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### API Key Setup

Create `.streamlit/secrets.toml` with:

```toml
GROQ_API_KEY = "your_groq_key"
```

Obtain from: [https://console.groq.com/playground](https://console.groq.com/playground)

---

## Usage

```bash
streamlit run main.py
```

Then:

* Enter candidate details
* Upload resume (PDF or DOCX)
* Receive questions from chatbot
* Respond and track confidence/sentiment
* Download final evaluation report

---

## Deployment (Streamlit Cloud)

1. Push code to public GitHub repo
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **Deploy App**
4. Choose your GitHub repo and branch
5. Add `GROQ_API_KEY` under **App Settings → Secrets**
6. Deploy — your app will be live shortly

---

## Steps Taken During Development

1. **Initial Setup:** Cloned template, structured Streamlit layout
2. **Input Validation:** Added `validators.py` for emails, phone, etc.
3. **Resume Parsing:** Integrated PDF/DOCX reading and claim validation
4. **Chatbot Logic:** Programmed context-aware Q\&A logic
5. **LLM Integration:** Connected to Groq Llama 3.3 70B API
6. **Dynamic Questions:** Added tech stack-based query generation
7. **Sentiment Analysis:** Introduced TextBlob evaluation for answer tone
8. **Multilingual Pipeline:** Enabled auto-detection and translation
9. **Feedback Module:** Built JSON-based reporting pipeline
10. **Streamlit Deployment:** Finalized .streamlit/secrets.toml and pushed to Streamlit Cloud

---

## Future Enhancements

* Resume plagiarism & AI-content detection
* Webcam/microphone-based proctoring
* Integrated code editor with compilation support
* Countdown timer for timed assessments
* Admin dashboard for analytics
* Candidate accounts and login system

---

## License

MIT License © 2025 TalentScout AI Team

---

## Acknowledgements

* Groq Llama 3.3 70B
* Streamlit
* TextBlob
* Langdetect
* PyPDF2 & python-docx
* Open Source Contributors

Live Demo: https://drive.google.com/file/d/1TKQ0YPOhV2Q6dIWfPU8CuK3uaZPDJGgX/view?usp=drive_link 

GitHub Repo: https://github.com/Rahul-Sanskar/TalentScout 

Streamlit: https://talentscout-qltp4z2dmudcy5bc8fxcmw.streamlit.app/ 
