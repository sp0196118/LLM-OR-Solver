"""
LLM-Powered OR Solver — Chat to Optimize
==========================================
User describes a business problem in plain English.
LLM parses it into a structured LP/ILP specification,
code is generated + executed with PuLP, result explained back.

Usage:
  python llm_or_solver.py          # CLI mode
  streamlit run app.py             # Interactive chat UI
"""

import json
import re
import textwrap
import pulp
import os

# ─────────────────────────────────────────────
# 1. LLM CLIENT  (Anthropic Claude)
# ─────────────────────────────────────────────
try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[WARN] anthropic package not installed. Using demo mode.")

SYSTEM_PROMPT = """You are an expert Operations Research solver assistant.
When a user describes a business optimization problem, you MUST respond with valid JSON only (no markdown, no explanation) in this exact format:

{
  "problem_type": "LP" | "ILP" | "MIP",
  "objective": "minimize" | "maximize",
  "objective_description": "brief description",
  "variables": [
    {"name": "x1", "description": "...", "type": "continuous" | "integer" | "binary", "lb": 0, "ub": null}
  ],
  "constraints": [
    {"description": "...", "expression": "2*x1 + 3*x2 <= 100", "label": "resource_A"}
  ],
  "objective_expression": "5*x1 + 4*x2",
  "plain_english_summary": "We are maximising profit from two products..."
}

Rules:
- Variable names must be simple: x1, x2, y1, etc.
- Expressions use *, +, -, <=, >=, ==
- Always include non-negativity as part of variable lb (set lb=0)
- If problem is infeasible as described, still structure it and note in summary
"""

def call_llm(user_problem: str) -> dict:
    """Send problem to LLM, get structured OR spec back."""
    if not LLM_AVAILABLE:
        return _demo_spec()

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_problem}]
    )
    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

def _demo_spec():
    """Demo spec used when no API key is present."""
    return {
        "problem_type": "LP",
        "objective": "maximize",
        "objective_description": "Maximise profit from two products",
        "variables": [
            {"name": "x1", "description": "Units of Product A", "type": "continuous", "lb": 0, "ub": None},
            {"name": "x2", "description": "Units of Product B", "type": "continuous", "lb": 0, "ub": None},
        ],
        "constraints": [
            {"description": "Labour hours limit", "expression": "2*x1 + 4*x2 <= 100", "label": "labour"},
            {"description": "Machine hours limit", "expression": "3*x1 + 2*x2 <= 90",  "label": "machine"},
            {"description": "Demand cap Product A", "expression": "x1 <= 30",           "label": "demand_A"},
        ],
        "objective_expression": "5*x1 + 7*x2",
        "plain_english_summary": "Demo: maximise profit (₹5/unit A + ₹7/unit B) subject to labour, machine and demand constraints."
    }

# ─────────────────────────────────────────────
# 2. DYNAMIC PULP SOLVER
# ─────────────────────────────────────────────
def solve_from_spec(spec: dict) -> dict:
    """Build and solve a PuLP model from the LLM-generated spec."""
    sense = pulp.LpMaximize if spec["objective"] == "maximize" else pulp.LpMinimize
    prob  = pulp.LpProblem("LLM_OR", sense)

    # Create variables
    var_map = {}
    for v in spec["variables"]:
        cat = {"continuous": "Continuous", "integer": "Integer", "binary": "Binary"}.get(v["type"], "Continuous")
        lb  = v.get("lb", 0)
        ub  = v.get("ub", None)
        var_map[v["name"]] = pulp.LpVariable(v["name"], lowBound=lb, upBound=ub, cat=cat)

    # Parse expression helper
    def parse_expr(expr_str: str):
        """Convert string expression to PuLP affine expression."""
        # Replace variable names with var_map references
        expr_str = expr_str.strip()
        # Use eval with var_map in scope (safe here since LLM output is structured)
        try:
            result = eval(expr_str, {"__builtins__": {}}, var_map)
            return result
        except Exception as e:
            raise ValueError(f"Cannot parse expression '{expr_str}': {e}")

    # Objective
    obj_expr = parse_expr(spec["objective_expression"])
    prob += obj_expr, "Objective"

    # Constraints
    for c in spec["constraints"]:
        expr_str = c["expression"]
        # Split on <=, >=, ==
        if "<=" in expr_str:
            lhs_str, rhs_str = expr_str.split("<=")
            prob += parse_expr(lhs_str.strip()) <= float(rhs_str.strip()), c["label"]
        elif ">=" in expr_str:
            lhs_str, rhs_str = expr_str.split(">=")
            prob += parse_expr(lhs_str.strip()) >= float(rhs_str.strip()), c["label"]
        elif "==" in expr_str:
            lhs_str, rhs_str = expr_str.split("==")
            prob += parse_expr(lhs_str.strip()) == float(rhs_str.strip()), c["label"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    solution = {v.name: pulp.value(v) for v in prob.variables() if not v.name.startswith("__")}
    obj_val  = pulp.value(prob.objective)

    return {
        "status": status,
        "objective_value": round(obj_val, 4) if obj_val else None,
        "variables": {k: round(v, 4) if v else 0 for k, v in solution.items()},
    }

# ─────────────────────────────────────────────
# 3. EXPLAIN RESULT IN PLAIN ENGLISH
# ─────────────────────────────────────────────
def explain_result(spec: dict, result: dict) -> str:
    if result["status"] != "Optimal":
        return f"⚠️ The solver returned status: {result['status']}. The problem may be infeasible or unbounded."

    obj_word = "maximum" if spec["objective"] == "maximize" else "minimum"
    lines = [
        f"✅ Optimal solution found!",
        f"",
        f"📌 Objective ({obj_word}): **{result['objective_value']}**",
        f"",
        f"📊 Decision Variables:",
    ]
    var_desc = {v["name"]: v["description"] for v in spec["variables"]}
    for name, val in result["variables"].items():
        desc = var_desc.get(name, name)
        lines.append(f"  • {desc} ({name}) = **{val}**")
    lines.append(f"")
    lines.append(f"💡 Summary: {spec['plain_english_summary']}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 4. CLI MODE
# ─────────────────────────────────────────────
EXAMPLE_PROBLEMS = [
    "A factory produces chairs and tables. Each chair requires 2 hours of labour and 3 kg of wood. Each table requires 4 hours of labour and 5 kg of wood. Labour available: 100 hours. Wood available: 150 kg. Profit: ₹500 per chair, ₹800 per table. Maximise total profit.",
    "A diet plan must include at least 50g protein, 30g fat, and 200g carbs. Food A costs ₹10 per 100g and provides 20g protein, 5g fat, 10g carbs per serving. Food B costs ₹15 per 100g with 10g protein, 15g fat, 30g carbs. Minimise total diet cost.",
    "A logistics company has 3 trucks (capacity 10, 8, 6 tonnes). Three deliveries of 4, 7, 5 tonnes must be assigned (one delivery per truck). Delivery costs: truck1→d1=100, truck1→d2=150, truck1→d3=120, truck2→d1=130, truck2→d2=110, truck2→d3=140, truck3→d1=160, truck3→d2=120, truck3→d3=100. Minimise total cost.",
]

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM-Powered OR Solver")
    print("=" * 60)
    print("\nExample problems (or type your own):")
    for i, p in enumerate(EXAMPLE_PROBLEMS, 1):
        print(f"\n[{i}] {textwrap.fill(p, 70, subsequent_indent='    ')}")

    print("\n" + "-"*60)
    choice = input("Enter 1/2/3 to use an example, or type your own problem:\n> ").strip()

    if choice in ["1","2","3"]:
        user_input = EXAMPLE_PROBLEMS[int(choice)-1]
    else:
        user_input = choice

    print(f"\n[USER] {user_input}")
    print("\n[LLM]  Parsing problem...")
    spec = call_llm(user_input)
    print(f"       Type: {spec['problem_type']} | Objective: {spec['objective']}")
    print(f"       Variables: {[v['name'] for v in spec['variables']]}")
    print(f"       Constraints: {len(spec['constraints'])}")

    print("\n[PuLP] Solving...")
    result = solve_from_spec(spec)
    print(f"       Status: {result['status']}")

    print("\n" + "="*60)
    print(explain_result(spec, result))
    print("="*60)

    # Save spec + result
    with open("last_solution.json", "w") as f:
        json.dump({"spec": spec, "result": result}, f, indent=2)
    print("\n[DONE] Full spec + result saved → last_solution.json")
