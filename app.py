"""
Streamlit Chat UI — LLM-Powered OR Solver
Run: streamlit run app.py
Set ANTHROPIC_API_KEY environment variable first.
"""
import streamlit as st
import json
import os
from llm_or_solver import call_llm, solve_from_spec, explain_result

st.set_page_config(page_title="OR Solver Chatbot", layout="centered")
st.title("🤖 LLM-Powered Operations Research Solver")
st.markdown("Describe any optimization problem in plain English. I'll parse, solve, and explain it.")

# API key input
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Anthropic API Key", type="password",
                            value=os.environ.get("ANTHROPIC_API_KEY",""),
                            help="Get yours at console.anthropic.com")
    print(api_key)
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    st.markdown("---")
    st.markdown("**Example problems:**")
    examples = [
        "Maximise profit making chairs (₹500) and tables (₹800). Labour: 100hrs total. Chairs need 2hrs, tables 4hrs.",
        "Minimise cost of diet using Food A (₹10/serving, 20g protein) and Food B (₹15/serving, 10g protein). Need 50g protein/day.",
        "3 workers, 3 tasks. Assign one each. Costs: w1-t1=10,w1-t2=8,w1-t3=6,w2-t1=5,w2-t2=7,w2-t3=9,w3-t1=4,w3-t2=6,w3-t3=8. Minimise total.",
    ]
    for i, ex in enumerate(examples, 1):
        if st.button(f"Example {i}", key=f"ex{i}"):
            st.session_state["prefill"] = ex

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Prefill from example button
prefill = st.session_state.pop("prefill", None)

user_input = st.chat_input("Describe your optimization problem...") or prefill
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        st.markdown(api_key)

    with st.chat_message("assistant"):
        with st.spinner("Parsing problem with LLM..."):
            try:
                spec = call_llm(user_input, api_key)
            except Exception as e:
                st.error(f"LLM parsing failed: {e}")
                st.stop()

        st.markdown(f"**Problem type**: `{spec['problem_type']}` | **Goal**: `{spec['objective']}`")

        with st.expander("📐 Parsed Model Structure"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Variables**")
                for v in spec["variables"]:
                    st.markdown(f"- `{v['name']}` ({v['type']}): {v['description']}")
            with col2:
                st.markdown("**Constraints**")
                for c in spec["constraints"]:
                    st.markdown(f"- `{c['expression']}` — {c['description']}")
            st.markdown(f"**Objective**: `{spec['objective_expression']}`")

        with st.spinner("Solving with PuLP..."):
            try:
                result = solve_from_spec(spec)
            except Exception as e:
                st.error(f"Solver error: {e}")
                st.stop()

        explanation = explain_result(spec, result)
        st.markdown(explanation)

        if result["status"] == "Optimal":
            import plotly.graph_objects as go
            var_names = list(result["variables"].keys())
            var_vals  = list(result["variables"].values())
            fig = go.Figure(go.Bar(x=var_names, y=var_vals, marker_color="#5364FF"))
            fig.update_layout(title="Optimal Variable Values", xaxis_title="Variable",
                              yaxis_title="Value", template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.messages.append({"role": "assistant", "content": explanation})
