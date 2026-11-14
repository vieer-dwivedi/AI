from node import (
    extract_info,
    match_score,
    generate_interview_questions,
    generate_feedback,
    rejection_message
)

# Sample input state
state = {
    "resume": """
    DevOps Engineer with 5 years of experience.
    Skills: AWS, Docker, Kubernetes, Terraform, Jenkins.
    """,

    "job_description": """
    Looking for a DevOps Engineer with AWS,
    Kubernetes, Docker and Terraform experience.
    """
}


# Test 1: Extract information
print("\n========== EXTRACT INFO ==========")
result = extract_info(state)
print("\n\n",result)
# Update state
state.update(result)


# Test 2: Match score
print("\n========== MATCH SCORE ==========")
result = match_score(state)
print("\n\n",result)

state.update(result)


# Test 3: Interview questions
print("\n========== INTERVIEW QUESTIONS ==========")
result = generate_interview_questions(state)
print("\n\n",result)

state.update(result)


# Test 4: Feedback
print("\n========== FEEDBACK ==========")
result = generate_feedback(state)
print("\n\n",result)

state.update(result)


# Test 5: Rejection message
print("\n========== REJECTION MESSAGE ==========")
result = rejection_message(state)
print("\n\n",result)

state.update(result)


print("\n========== FINAL STATE ==========")
print("\n\n",state)