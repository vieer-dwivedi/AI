from app.agents.python_agent import PythonAgent

AGENTS = {"python_executor": PythonAgent}

def dispatch(task):
    agent = AGENTS[task["agent_type"]]()
    return agent.execute(task)
