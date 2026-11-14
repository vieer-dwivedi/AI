import os


def validate_input(state):

    errors = []

    resume_file = state.get("resume_file")
    job_description = state.get("job_description", "").strip()

    # Validate resume path
    if not resume_file:
        errors.append("Resume file path is missing.")

    elif not os.path.exists(resume_file):
        errors.append(f"Resume file not found: {resume_file}")

    else:
        extension = os.path.splitext(resume_file)[1].lower()

        if extension not in [".pdf", ".docx", ".txt"]:
            errors.append("Unsupported resume format. Use PDF, DOCX, or TXT.")

    # Validate job description
    if not job_description:
        errors.append("Job description cannot be empty.")

    if errors:
        raise ValueError("\n".join(errors))

    return state