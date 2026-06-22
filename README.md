# 🤖 LLM-Powered OR Solver — Chat to Optimize

> Describe any business optimization problem in plain English. Claude parses it into a structured LP/ILP, solves it with PuLP, and explains the result back in plain language.

## 🎯 Problem Statement
Most people can describe what they want to optimise but can't write mathematical models. This project bridges that gap — natural language → LP/ILP → optimal solution → plain-English explanation.

## 🏗️ Architecture
```
User describes problem in plain English
        │
        ▼
  Claude LLM                        ← AI Layer
  Parses → structured JSON spec
  {variables, constraints, objective}
        │
        ▼
  Dynamic PuLP Model Builder         ← OR Layer
  (builds + solves LP/ILP at runtime)
        │
        ▼
  Result explained back in plain language
  + bar chart of optimal values
```

## 📦 Tech Stack
| Layer | Tool |
|-------|------|
| LLM | `anthropic` Claude claude-sonnet-4-6 |
| OR | `PuLP` CBC solver (LP + ILP + MIP) |
| UI | `streamlit` chat interface |
| Visualisation | `plotly` |

## 🚀 Quick Start
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here

# CLI mode
python llm_or_solver.py

# Chat UI
streamlit run app.py
```

## 💬 Example Conversations
**User**: "A factory makes chairs (₹500 profit, 2 labour hours each) and tables (₹800 profit, 4 hours each). I have 100 hours of labour. Maximise profit."

**Solver**: Optimal! Produce 50 chairs + 0 tables = ₹25,000 profit. Or with a wood constraint...

**User**: "Assign 3 workers to 3 tasks to minimise total cost. Costs: [matrix]"

**Solver**: Assignment problem solved. Optimal cost = ₹21. Worker 1→Task 3, Worker 2→Task 1, Worker 3→Task 2.

## 🔑 What Makes This Unique
- **No template** — solves arbitrary LP/ILP problems described in free text
- **Live code generation** — PuLP model is built dynamically from LLM output, not hardcoded
- **Explainable** — result explained in same language as the question
- **Extensible** — add more solvers (GLPK, Gurobi) or problem types (multi-objective, stochastic)

## 💼 Interview Talking Points
1. How do you prevent LLM hallucination in the model spec? *(strict JSON schema + validation + fallback error handling)*
2. What problem types can this handle? *(LP, ILP, MIP, assignment, transportation, knapsack — anything expressible in PuLP)*
3. How would you productionise this? *(FastAPI endpoint, rate limiting, model spec caching, Gurobi for large instances)*

## 📁 Files
```
05_llm_or_solver/
├── llm_or_solver.py   # Core pipeline + CLI
├── app.py             # Streamlit chat UI
├── requirements.txt
└── README.md
```
