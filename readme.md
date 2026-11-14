# 🤖 Job Application Screener Agent

An AI-powered resume screening system built with **LangGraph** and **LangChain** that automatically evaluates candidates against job descriptions and routes them through intelligent decision pipelines.

---

## 📌 What This Project Does

This agent takes a **resume file** (PDF, DOCX, or TXT) and a **job description** as input, then:

1. Extracts text from the resume file
2. Uses an LLM to identify skills and years of experience
3. Scores the candidate against the job description (0–100)
4. Routes to one of three outcomes based on the score
5. Saves a structured screening report to a file

---

## 🔄 Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                │
│         resume_file + job_description                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     input_node                              │
│   Reads resume file → supports PDF / DOCX / TXT            │
│   Extracts raw text using PyMuPDF / python-docx             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    extract_info                             │
│   LLM extracts:                                             │
│     • extracted_skills  → list of skills                    │
│     • extracted_experience → years of experience            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    match_score                              │
│   LLM evaluates resume vs job description                   │
│     • match_score  → 0 to 100                               │
│     • match_reason → explanation                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   screening_decision  │  ← Conditional Edge
          │   (router function)   │
          └───────────┬───────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   score >= 80    score 50–79   score < 50
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   generate   │ │   generate   │ │  rejection   │
│  interview   │ │   feedback   │ │   message    │
│  questions   │ │              │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │       save_result       │
          │  Writes report to .txt  │
          └────────────┬────────────┘
                       │
                       ▼
                      END
```

---

## 🗂️ Project Structure

```
Job Application Screener Agent/
│
├── main.py                  # Entry point — invokes the graph
├── resume.pdf               # Sample resume for testing
├── screening_results.txt    # Output — appended after each run
├── .env                     # API keys (MODEL, GOOGLE_API_KEY)
│
└── core/
    ├── state.py             # AgentState (TypedDict)
    ├── graph.py             # Node registration + edge wiring
    ├── node.py              # All node functions + save logic
    ├── llm.py               # LLM initialization
    └── __init__.py
```

---

## 🧠 LangGraph Skills Demonstrated

### Graph Concepts
| Concept | Where Used |
|---|---|
| `StateGraph` | Core graph definition in `graph.py` |
| `TypedDict` state | `ScreenerState` in `state.py` |
| `NotRequired` fields | Optional state fields that only populate on certain branches |
| `set_entry_point` | Graph starts at `input` node |
| `add_edge` | Linear edges between sequential nodes |
| `add_conditional_edges` | 3-way routing from `match_score` node |
| `END` | Terminal node after `save_result` |
| `graph.compile()` | Compiled into runnable `program_agent` |

### Node Design
| Concept | Where Used |
|---|---|
| LLM inside a node | `extract_info`, `match_score`, all output nodes |
| File I/O in a node | `input_node` (read resume), `save_result` (write report) |
| JSON parsing helper | `parse_json()` strips markdown fences before parsing |
| State passing | Each node returns a dict that merges into global state |
| Multi-format file support | PDF via PyMuPDF, DOCX via python-docx, TXT native |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **LangGraph** | Graph orchestration, state management, routing |
| **LangChain** | LLM abstraction via `init_chat_model` |
| **Google Gemini** | LLM for all extraction and generation tasks |
| **PyMuPDF (fitz)** | PDF text extraction |
| **python-docx** | DOCX text extraction |
| **python-dotenv** | Environment variable management |
| **LangSmith** | Graph visualization and execution tracing |

---

## 🔀 Routing Logic

```python
def screening_decision(state: ScreenerState):
    score = state["match_score"]
    if score >= 80:
        return "high"    →  generate_interview_questions
    elif score >= 50:
        return "medium"  →  generate_feedback
    else:
        return "low"     →  rejection_message
```

---

## 📥 Input (LangSmith Studio)

```json
{
  "resume_file": "resume.pdf",
  "job_description": "DevOps engineer with Kubernetes and AWS experience"
}
```

---

## 📤 Output

A `screening_results.txt` report is appended after every run:

```
======================================================================
                 RESUME SCREENING REPORT
======================================================================
Date: 2026-11-14 14:49:23
Resume: resume.pdf
Experience: 5 years
Match Score: 80/100
Decision: Interview

----------------------------------------------------------------------
EXTRACTED SKILLS
----------------------------------------------------------------------
- Python, AWS, Kubernetes, Terraform ...

----------------------------------------------------------------------
MATCH REASON
----------------------------------------------------------------------
Strong cloud and DevOps background matching all core requirements.

----------------------------------------------------------------------
INTERVIEW QUESTIONS
----------------------------------------------------------------------
1. Describe your experience with EKS cluster management...
======================================================================
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install langgraph langchain python-dotenv pymupdf python-docx

# 2. Set up .env
MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_key_here

# 3. Run
python main.py
```

---

## 🔮 Planned Improvements

- [ ] Add `validate_input` node before extraction
- [ ] Add `human_review` node using `interrupt()` for borderline scores (65–75)
- [ ] Add retry loop if LLM returns invalid JSON
- [ ] Add `decision` field to state
- [ ] Batch screening — process a folder of resumes in parallel using `Send` API
- [ ] Align score thresholds between `graph.py` and `save_result`

---

## 👤 Author

Built by **Vieer** as an intermediate LangGraph learning project.  
Focus: mastering nodes, conditional edges, state design, and multi-branch graph routing.