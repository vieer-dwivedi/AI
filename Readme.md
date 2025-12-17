
```bash
┌────────────────────┐
│   User / Client    │
└─────────┬──────────┘
          │ instruction
          ▼
┌────────────────────────────┐
│   Master Agent Service     │
│  (Planner / Orchestrator) │
│                            │
│ - Validates input           │
│ - Asks LLM to plan tasks    │
│ - Splits into ordered tasks │
│ - Adds context per task     │
└─────────┬──────────────────┘
          │ enqueue (tasks)
          ▼
┌────────────────────────────┐
│       Azure Queue          │
│  (Reliable Task Buffer)   │
│                            │
│ - One message = one task   │
│ - Visibility timeout       │
│ - At-least-once delivery   │
└─────────┬──────────────────┘
          │ poll
          ▼
┌─────────────────────────────────────────┐
│      Executor / Slave Agent Service     │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Queue Consumer                    │  │
│  │ - Polls Azure Queue               │  │
│  │ - Deletes on success              │  │
│  └───────────────┬───────────────────┘  │
│                  │ dispatch             │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │ Dispatcher                        │  │
│  │ - Routes by agent_type            │  │
│  └───────────────┬───────────────────┘  │
│                  │                      │
│      ┌───────────▼───────────┐          │
│      │   Python Agent        │          │
│      │  (Slave Agent)        │          │
│      │                       │          │
│      │ - Writes files        │          │
│      │ - Runs commands/tests │          │
│      │ - Detects errors      │          │
│      └───────────┬───────────┘           │
│                  │ (on failure)          │
│                  ▼                       │
│        ┌──────────────────────────┐      │
│        │        LLM Service       │      │
│        │ (shared via HTTP API)    │      │
│        │                          │      │
│        │ - Sees file              │      │
│        │ - Sees stdout/stderr     │      │
│        │ - Sees acceptance rules  │      │
│        │ - Returns fixed content  │      │
│        └──────────────────────────┘      │
│                                          │
│ - Retries up to N times                  │
│ - ACKs queue only on success             │
└──────────────────────────────────────────┘
```

# 🤖 Agentic AI Platform – Master & Slave Agent Architecture

Welcome!  
This project demonstrates a **production-grade Agentic AI system** built on a **clear separation of reasoning and execution**, backed by **queues** and a **shared LLM service**.

> **Think → Plan → Execute → Validate → Fix → Repeat (if needed)**

---

## 🧠 What Is This?

This system converts a **single high-level user instruction** into:
- A structured plan
- Multiple executable tasks
- Reliable execution
- Automated fixing using an LLM (only on failure)

It follows a **Master–Slave Agent model**, which is scalable, debuggable, and cloud-native.

---


---

## 🧩 Core Concepts

### 🔑 Workflow
A **workflow** represents one complete user request, identified by a unique `workflow_id`.

### 🔧 Task
A **task** is the smallest executable unit:
- Create a file
- Run a command
- Run tests

Each task is independent and queue-driven.

### 📜 Context
Each task carries:
- File goal
- Run instructions
- Test instructions
- Acceptance criteria

Nothing is assumed. Everything is explicit.

---

## 🤖 Types of Agents

---

### 🧠 Master Agent (Planner / Orchestrator)

**Role:** *Think & Plan*

**Responsibilities**
- Accepts high-level user instructions
- Asks the LLM to reason and plan
- Splits work into ordered tasks
- Adds rich execution context
- Pushes tasks to the queue

**Does NOT**
- Write files
- Execute code
- Run tests

> The Master Agent **thinks**, but never acts.

---

### ⚙️ Slave Agents (Executors / Workers)

**Role:** *Act & Validate*

**Responsibilities**
- Consume one task at a time from the queue
- Write files exactly as instructed
- Run commands and tests
- Validate acceptance criteria
- Retry safely if failures occur
- Acknowledge (delete) queue messages on success

**Does NOT**
- Plan future tasks
- Modify workflow structure
- Make business decisions

> Slave agents **act**, but never think beyond their task.

---

### 🧠 LLM Reasoning Agent (Shared)

**Role:** *Assist, Not Control*

The LLM is used **only when execution fails**.

It receives:
- File content
- Command executed
- STDOUT / STDERR
- Exit code
- Acceptance criteria

It returns:
- **Corrected file content only**

Strict rules prevent hallucination or side effects.

---

## 🔁 Execution Flow (Step-by-Step)

1. User submits an instruction
2. Master Agent plans tasks using the LLM
3. Tasks are queued (one message = one task)
4. Executor Service polls the queue
5. Correct slave agent executes the task
6. If execution fails:
   - LLM is invoked with full context
   - File is corrected
   - Task retries (bounded)
7. On success:
   - Message is deleted from the queue

---

## 🔒 Reliability & Safety

| Feature | How It’s Achieved |
|------|------------------|
| Fault tolerance | Queue visibility timeouts |
| No infinite loops | Max retry limits |
| Determinism | Acceptance criteria |
| Scalability | Multiple executor containers |
| Isolation | Workflow-scoped workdirs |

---

## 📂 Project Structure (Executor Service)


---

## 🚀 Why This Architecture Works

✅ Clean separation of concerns  
✅ Horizontally scalable  
✅ Debuggable and observable  
✅ Safe use of LLMs  
✅ Cloud-native design  
✅ Production-ready

---

## 🧠 Philosophy (Important)

> **LLMs reason.  
> Agents execute.  
> Queues coordinate.  
> Systems stay deterministic.**

---

## 🔮 What Can Be Added Next?

- Prometheus metrics
- Dead-letter queues
- Workflow dependency graphs
- Kubernetes auto-scaling
- Human-in-the-loop approvals
- More agent types (Node, Go, Terraform)

---

## 📌 Final Thought

This project is not a demo toy.  
It is a **real, extensible foundation** for building reliable Agentic AI systems.

If you understand this architecture, you understand how **modern AI platforms are actually built**.
