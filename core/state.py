from typing import TypedDict, NotRequired

class ScreenerState(TypedDict):
    save_status: str
    resume_file: str
    resume: str
    job_description: str
    extracted_skills: NotRequired[list[str]]
    extracted_experience: NotRequired[int]
    match_score: NotRequired[int]
    match_reason: NotRequired[str]
    interview_questions: NotRequired[list[str]]
    feedback: NotRequired[str]
    rejection_message: NotRequired[str]