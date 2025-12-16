def build_fix_prompt(task, stdout, stderr, exit_code):
    return f"""
You are an automated code-fixing agent.

RULES:
- Return ONLY corrected file content
- No markdown, no explanations

FILE:
{task['file']['path']}

CONTENT:
{task['file']['content']}

COMMAND:
{task['context'].get('run_instructions')}

EXIT CODE: {exit_code}

STDOUT:
{stdout}

STDERR:
{stderr}

GOAL:
{task['context'].get('file_goal')}

CRITERIA:
{chr(10).join(task['context'].get('acceptance_criteria', []))}

Fix the file so execution succeeds.
"""
