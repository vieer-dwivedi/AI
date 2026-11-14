from langgraph.graph import StateGraph, END
from core.state import ScreenerState
from core.middleware import validate_input

from core.node import (
    input_node,
    extract_info,
    match_score,
    generate_feedback,
    generate_interview_questions,
    rejection_message,
    save_result
)


def screening_decision(state: ScreenerState):
    score = state["match_score"]

    if score >= 80:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"


graph = StateGraph(ScreenerState)

# Register nodes
graph.add_node("input", input_node)
graph.add_node("validate_input", validate_input)
graph.add_node("extract_info", extract_info)
graph.add_node("match_score", match_score)
graph.add_node("generate_interview_questions", generate_interview_questions)
graph.add_node("generate_feedback", generate_feedback)
graph.add_node("rejection_message", rejection_message)
graph.add_node("save_result", save_result)

# Main flow
graph.set_entry_point("validate_input")
graph.add_edge("validate_input", "input")

graph.add_edge("input", "extract_info")
graph.add_edge("extract_info", "match_score")

# Conditional routing
graph.add_conditional_edges(
    "match_score",
    screening_decision,
    {
        "high": "generate_interview_questions",
        "medium": "generate_feedback",
        "low": "rejection_message"
    }
)

# Save every branch
graph.add_edge("generate_interview_questions", "save_result")
graph.add_edge("generate_feedback", "save_result")
graph.add_edge("rejection_message", "save_result")

graph.add_edge("save_result", END)

program_agent = graph.compile()