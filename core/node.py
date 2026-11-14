from core.state import ScreenerState
from core.llm import model
import json
from pathlib import Path
from datetime import datetime



def parse_json(content):
    if isinstance(content, list):
        content = "".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)


def save_result(state: ScreenerState):

    file_path = Path(__file__).resolve().parent.parent / "screening_results.txt"

    score = state.get("match_score", 0)

    # Determine screening decision
    if score > 75:
        decision = "Interview"
    elif 65 <= score <= 75:
        decision = "Human Review"
    elif score >= 50:
        decision = "Feedback"
    else:
        decision = "Rejected"

    skills = state.get("extracted_skills", [])
    questions = state.get("interview_questions", [])

    # Append mode: preserves previous screening results
    with open(file_path, "a", encoding="utf-8") as file:

        file.write("\n\n")
        file.write("=" * 70 + "\n")
        file.write("                 RESUME SCREENING REPORT\n")
        file.write("=" * 70 + "\n")

        file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Resume: {state.get('resume_file', 'N/A')}\n")
        file.write(f"Experience: {state.get('extracted_experience', 'N/A')} years\n")
        file.write(f"Match Score: {score}/100\n")
        file.write(f"Decision: {decision}\n")

        file.write("\n" + "-" * 70 + "\n")
        file.write("EXTRACTED SKILLS\n")
        file.write("-" * 70 + "\n")

        for skill in skills:
            file.write(f"- {skill}\n")

        file.write("\n" + "-" * 70 + "\n")
        file.write("MATCH REASON\n")
        file.write("-" * 70 + "\n")
        file.write(f"{state.get('match_reason', 'N/A')}\n")

        if questions:
            file.write("\n" + "-" * 70 + "\n")
            file.write("INTERVIEW QUESTIONS\n")
            file.write("-" * 70 + "\n")

            for index, question in enumerate(questions, 1):
                file.write(f"{index}. {question}\n")

        if state.get("feedback"):
            file.write("\nFEEDBACK\n")
            file.write("-" * 70 + "\n")
            file.write(f"{state['feedback']}\n")

        if state.get("rejection_message"):
            file.write("\nREJECTION MESSAGE\n")
            file.write("-" * 70 + "\n")
            file.write(f"{state['rejection_message']}\n")

        if state.get("human_review"):
            file.write("\nHUMAN REVIEW\n")
            file.write("-" * 70 + "\n")
            file.write(f"{state['human_review']}\n")
            file.write(
                f"Review Status: {state.get('review_status', 'Pending')}\n"
            )

        file.write("=" * 70 + "\n")

    return {
        "save_status": f"Result saved successfully to {file_path}"
    }
import os
import fitz
from docx import Document
def input_node(state: ScreenerState):

    file_path = state["resume_file"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        doc = fitz.open(file_path)
        resume_text = "\n".join(
            page.get_text() for page in doc
        )

    elif extension == ".docx":
        doc = Document(file_path)
        resume_text = "\n".join(
            p.text for p in doc.paragraphs
        )

    elif extension == ".txt":
        with open(file_path, encoding="utf-8") as f:
            resume_text = f.read()

    else:
        raise ValueError("Only PDF, DOCX and TXT supported")

    if not resume_text.strip():
        raise ValueError("Resume contains no extractable text")

    return {
        "resume": resume_text
    }

def extract_info(state: ScreenerState):
    response = model.invoke(
        f"""
        Extract skills and total professional experience
        from the resume.

        Return ONLY valid JSON:
        {{
            "extracted_skills": ["Python", "AWS"],
            "extracted_experience": 5
        }}

        Resume:
        {state['resume']}
        """
    )
    data = parse_json(response.content)
    return {
        "extracted_skills": data["extracted_skills"],
        "extracted_experience": data["extracted_experience"]
    }

def match_score(state: ScreenerState):
    response = model.invoke(
        f"""
        Evaluate the candidate against the job description.

        Return ONLY valid JSON:
        {{
            "match_score": 85,
            "match_reason": "Strong technical match"
        }}

        Resume:
        {state['resume']}

        Job Description:
        {state['job_description']}

        Skills:
        {state['extracted_skills']}

        Experience:
        {state['extracted_experience']} years

        Score between 0 and 100.
        Do not invent qualifications.
        """
    )
    data = parse_json(response.content)
    return {
        "match_score": data["match_score"],
        "match_reason": data["match_reason"]
    }

def generate_interview_questions(state: ScreenerState):
    response = model.invoke(
        f"""
        Generate exactly 5 interview questions.

        Return ONLY valid JSON:
        {{
            "questions": [
                "Question 1",
                "Question 2",
                "Question 3",
                "Question 4",
                "Question 5"
            ]
        }}

        Job Description:
        {state['job_description']}

        Skills:
        {state['extracted_skills']}

        Experience:
        {state['extracted_experience']}
        """
    )
    data = parse_json(response.content)
    return {
        "interview_questions": data["questions"]
    }

def generate_feedback(state: ScreenerState):

    response = model.invoke(
        f"""
        Provide actionable resume improvement feedback.

        Skills:
        {state['extracted_skills']}

        Match Score:
        {state['match_score']}

        Match Reason:
        {state['match_reason']}
        """
    )

    return {
        "feedback": response.content
    }

def rejection_message(state: ScreenerState):
    response = model.invoke(
        f"""
        Write a polite and professional rejection message.
        Match Score: {state['match_score']}/100
        Reason: {state['match_reason']}
        """
    )
    return {
        "rejection_message": response.content
    }

