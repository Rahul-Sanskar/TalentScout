# TalentScout Hiring Assistant Documentation

## Overview
TalentScout Hiring Assistant is a powerful AI-driven platform designed to help recruiters and hiring managers evaluate candidates with speed and precision. By leveraging advanced natural language processing, dynamic question generation, and a user-friendly interface, the system seamlessly balances functionality and scalability. This documentation aims to provide a comprehensive look at TalentScout—from its file structure and modular design to the core features that enable efficient candidate assessment.

---

## Table of Contents
1. [System Architecture](#system-architecture)  
2. [Features](#features)  
    * [Functionality](#functionality)  
    * [User Interface (UI)](#user-interface-ui)  
    * [Chatbot Capabilities](#chatbot-capabilities)  
    * [Resume Analysis](#resume-analysis)  
    * [Dynamic Question Generation](#dynamic-question-generation)  
    * [Confidence Scoring](#confidence-scoring)  
    * [Feedback and Reporting](#feedback-and-reporting)  
    * [Advanced Prompt Engineering](#advanced-prompt-engineering)  
    * [State Management](#state-management)  
3. [File Structure](#file-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Future Enhancements](#future-enhancements)

---

## System Architecture
TalentScout is built upon a modular architecture to ensure maintainability and scalability. Each major component is encapsulated in a dedicated module or folder. This encapsulation allows easy updates, clear coding structure, and reusability across different functionalities.

### Modular Design
1. **Validation and Resume Processing**  
    - Handles user input validation and resume parsing.  
    - Uses regex extensively to check candidate information (e.g., phone number, email).  
2. **User Interface**  
    - Developed with Streamlit for an intuitive experience.  
    - Structured layout ensures clarity for both candidates and administrators.  
3. **Assessment and Scoring**  
    - Dynamically generates questions based on experience, tech stack, and prior answers.  
    - Maintains a confidence score that adjusts after every response.  
4. **Reporting**  
    - Facilitates feedback generation and downloadable reports.  
    - Highlights strengths, weaknesses, and possible areas for improvement.  
5. **LLM Integration**  
    - Tightly integrated with the Llama 3.3 70B Versatile model for fast, contextually relevant responses.  
    - Utilizes advanced prompt engineering for improved question accuracy.

---

## Features

### Functionality
- **Dynamic Processing Pipelines**  
  Built to handle user inputs, process resumes, and generate tailored questions in real time.
- **Extensive Comments and Code Clarity**  
  Maintainers can easily modify or extend functionalities without confusion.
- **High Scalability**  
  The loosely coupled design makes it straightforward to add more models, question types, or data sources.
- **Efficient Resume-Data Linking**  
  Compares user-declared positions, experience, and tech stack with resume data to prevent mismatches.

---

### User Interface (UI)
- **Streamlit-based**  
  Presents a clean, browser-accessible interface for candidate interaction.
- **Sidebar Guidance**  
  Displays guidelines, motivational quotes, and rules candidates must follow.
- **Simple Data Collection**  
  Collects personal details (name, email, phone, location, years of experience) via interactive fields.
- **Resume Upload**  
  Allows PDF and DOCX uploads for quick analysis.
- **Early Termination**  
  Candidates can end the assessment anytime. The system also auto-ends if too many questions are skipped.

---

### Chatbot Capabilities

#### Greeting
- Greets candidates when they enter the system.  
- Provides a brief overview of its operation and objectives.

#### Information Gathering
- Prompts for essential details such as:  
  - Full Name  
  - Email Address  
  - Phone Number  
  - Years of Experience  
  - Desired Position(s)  
  - Current Location  
  - Tech Stack

#### Tech Stack Declaration
- Encourages the candidate to list programming languages, frameworks, databases, and tools.
- Uses this data to tailor future interactions.

#### Context Handling
- Maintains context throughout the conversation to ensure timely and relevant follow-up questions.

#### Fallback Mechanism
- Offers informative responses if user input is unclear or unexpected.
- Ensures the conversation remains coherent and goal-directed.

#### Conversation Ending
- Gracefully closes the session on a recognized keyword or after excessive question-skipping.
- Thanks the candidate and indicates possible next steps.

---

### Resume Analysis
- **Automatic Parsing**  
  Uses PyPDF2 and python-docx to extract text from PDF and DOCX resumes.  
- **Claim Validation**  
  Aligns the user’s stated experience, tech stack, and desired role with resume details.  
- **Discrepancy Checks**  
  Flags irrelevant or unsupported claims.  
- **Consistency Score**  
  Helps recruiters see how closely a candidate’s self-assessed abilities match their provided documentation.

---

### Dynamic Question Generation
- **Contextual Depth**  
  Starts with easier questions, increases complexity based on candidate responses.  
- **Tech Stack Relevance**  
  If a user declares Python and Django, relevant Python/Django questions are generated.  
- **Adaptability**  
  Incorporates the confidence score and previous correct/incorrect answers to refine future questions.  
- **Advanced Prompt Engineering**  
  Uses specialized queries to the Llama 3.3 70B Versatile model for relevant, targeted questions.

---

### Confidence Scoring
- **Real-time Adjustments**  
  Increases if users perform well on complex prompts and decreases for incorrect or skipped simple questions.
- **Holistic Assessment**  
  Balances technical correctness, clarity, and completeness of answers.  
- **Influence on Flow**  
  Helps the system decide whether to pose more challenging questions or revert to fundamentals.

---

### Feedback and Reporting
- **Detailed Summaries**  
  Outlines the user’s performance, highlighting strong and weak points.  
- **Downloadable Formats**  
  Enables JSON or text export for later review.  
- **Progress Visibility**  
  Displays confidence scores and significant metrics in real time.  
- **Actionable Insights**  
  Suggests areas of further learning or preparation based on question performance.

---

### Advanced Prompt Engineering
- **Precision Targeting**  
  Generates relevant questions aligned with the user’s resume and inputs without drifting off-topic.
- **Multiple Rounds**  
  The conversation can span multiple turns, with advanced memory of prior answers.  
- **Seamless LLM Integration**  
  Llama 3.3 70B Versatile from Groq handles large request volumes with speed and accuracy.

---

### State Management
- **Efficient Caching**  
  Minimizes redundant LLM calls and ensures quick load times.  
- **Session Persistence**  
  Retains user context (answers, confidence score, etc.) between steps in the conversation.  
- **Scalability**  
  Can easily integrate additional caching layers or switch to distributed session stores if needed.

---

## File Structure
Below is a suggested layout for clarity and maintainability:

```
project/
│
├── main.py
├── utils/
│   ├── validators.py
│   ├── resume_processing.py
├── components/
│   ├── sidebar.py
│   ├── progress.py
├── assessment/
│   ├── question_generation.py
│   ├── evaluation.py
├── config/
│   ├── settings.py
├── models/
│   ├── llm_manager.py
├── reporting/
│   ├── report_generator.py
```

- **main.py**  
  Orchestrates the Streamlit app, manages high-level flow.  
- **utils/validators.py**  
  Contains regex-based and logical validations for user inputs.  
- **utils/resume_processing.py**  
  Handles file loading and text extraction from resumes.  
- **components/sidebar.py**  
  Manages the Streamlit sidebar elements, including guidelines and quotes.  
- **components/progress.py**  
  Displays progress or confidence-related metrics in the UI.  
- **assessment/question_generation.py**  
  Creates context-based questions using Llama 3.3 70B Versatile model.  
- **assessment/evaluation.py**  
  Rates user responses and updates the confidence score.  
- **config/settings.py**  
  Stores environment variables and API keys (secured with secrets).  
- **models/llm_manager.py**  
  Provides an interface to the Llama 3.3 70B Versatile model for queries.  
- **reporting/report_generator.py**  
  Aggregates user performance data and generates final reports.

---

## Installation

### Prerequisites
- Python 3.8 or higher  
- Pip  
- Internet connection

### Steps
1. **Clone the Repository**  
    ```bash
    git clone https://github.com/your-repo/talentscout.git
    cd talentscout
    ```
2. **Create a Virtual Environment**  
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3. **Install Dependencies**  
    ```bash
    pip install -r requirements.txt
    ```
4. **Configure Streamlit Secrets**  
    ```bash
    echo "GROQ_API_KEY=your_api_key" > .streamlit/secrets.toml
    ```
5. **Run the Application**  
    ```bash
    streamlit run main.py
    ```

---

## Usage

1. **Launch the Application**  
    ```bash
    streamlit run main.py
    ```
2. **Enter Personal Details**  
    Fill in name, email, phone number, years of experience, and desired roles.
3. **Upload Resume (PDF or DOCX)**  
    The system parses relevant content for verification.
4. **Interact with the Chatbot**  
    - Answer dynamically generated technical questions.  
    - Check the sidebar for guiding rules and motivational quotes.
5. **Review Performance**  
    - Monitor your evolving confidence score.  
    - See flagged discrepancies or missing skills.  
6. **End and Feedback**  
    - Conclude at will or let the system auto-end if too many questions are skipped.  
    - Download your performance report in JSON or text format.

---

## Future Enhancements
1. **AI Content Detection**  
    Identify plagiarized or AI-generated content in resumes.  
2. **Camera and Mic Monitoring**  
    Incorporate proctoring features for remote assessments.  
3. **Countdown Timer**  
    Keep a timed environment to simulate real interview pressure.  
4. **Code Compiler Integration**  
    Allow in-app coding exercises for more hands-on tests.  
5. **User Account Creation**  
    Let candidates log in and track their progress or history over time.

