from core.graph import program_agent
import json

result = program_agent.invoke({
    "resume_file": "resume.pdf",
    "job_description": "devops engineer with kuberneets and Flutter developer with dart and android exp"
})

print("\n" + "=" * 60)
print("           RESUME SCREENING REPORT")
print("=" * 60)

print(f"\nMatch Score: {result.get('match_score', 'N/A')}/100")
print(f"Experience: {result.get('extracted_experience', 'N/A')} years")

print("\n" + "-" * 60)
print("EXTRACTED SKILLS")
print("-" * 60)

for skill in result.get("extracted_skills", []):
    print(f"  • {skill}")

print("\n" + "-" * 60)
print("MATCH REASON")
print("-" * 60)
print(result.get("match_reason", "Not available"))

if result.get("interview_questions"):
    print("\n" + "-" * 60)
    print("INTERVIEW QUESTIONS")
    print("-" * 60)

    for i, question in enumerate(result["interview_questions"], 1):
        print(f"{i}. {question}")

if result.get("feedback"):
    print("\n" + "-" * 60)
    print("CANDIDATE FEEDBACK")
    print("-" * 60)
    print(result["feedback"])

if result.get("rejection_message"):
    print("\n" + "-" * 60)
    print("SCREENING RESULT")
    print("-" * 60)
    print(result["rejection_message"])

print("\n" + "=" * 60)
print("             SCREENING COMPLETED")
print("=" * 60)