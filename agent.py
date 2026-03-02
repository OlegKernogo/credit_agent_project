import json
from typing import Dict, Any, Literal, Optional, List   
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from models import ProcessedMetrics, CreditApplication, CreditDecision
from typing_extensions import TypedDict

# Define Graph State
class AgentState(TypedDict):
    metrics: ProcessedMetrics
    application: CreditApplication
    route: str  # "simple", "debate", "tools"
    reasoning: str
    is_approved: bool
    recommended_amount: Optional[float]
    recommended_term: Optional[int]
    confidence_score: float
    guardrail_passed: bool
    policy_violations: List[str]
    guardrail_triggers: List[str]
    decision: CreditDecision

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Nodes
def supervisor_node(state: AgentState):
    """Determines which reasoning strategy to use."""
    metrics = state["metrics"]
    app = state["application"]
    
    # If requested amount is very small compared to average cash, simple is fine.
    if app.requested_amount < (metrics.average_cash_balance * 0.5):
        route = "simple"
    elif app.requested_amount > metrics.fixed_assets_volume * 0.8:
        # High-risk: requesting more than 80% of fixed asset collateral
        route = "debate"
    else:
        # Standard processing
        route = "tools"
        
    return {"route": route}

def reasoning_simple_node(state: AgentState):
    """Simple immediate decision based on basic thresholds."""
    metrics = state["metrics"]
    app = state["application"]
    
    reasoning = (
        f"Used Simple Reasoning. Requested {app.requested_amount} {app.currency}. "
        f"Average Cash: {metrics.average_cash_balance}. "
        f"Cash covers loan easily."
    )
    is_approved = True
    confidence = 0.95
        
    return {"reasoning": reasoning, "is_approved": is_approved, "confidence_score": confidence, "recommended_amount": app.requested_amount, "recommended_term": app.loan_term_months}

def reasoning_debate_node(state: AgentState):
    """Uses LLM to debate pros and cons of the high-risk loan."""
    metrics = state["metrics"]
    app = state["application"]
    
    sys_msg = SystemMessage(content="You are a strict bank credit committee. Debate the pros and cons of approving this loan based on the financial metrics. Conclude with APPROVED or REJECTED and a confidence score between 0.0 and 1.0.")
    user_msg = HumanMessage(content=f"Loan Term: {app.loan_term_months} months. Amount: {app.requested_amount} {app.currency}. Purpose: {app.purpose}\nMetrics: {json.dumps(metrics.dict())}")
    
    response = llm.invoke([sys_msg, user_msg])
    text = response.content
    
    is_approved = "APPROVED" in text.upper()
    reasoning = f"Debated: {text}"
    confidence = 0.7 
    
    return {"reasoning": reasoning, "is_approved": is_approved, "confidence_score": confidence, "recommended_amount": app.requested_amount if is_approved else 0, "recommended_term": app.loan_term_months if is_approved else 0}

def reasoning_tools_node(state: AgentState):
    """Standard evaluation combining metrics and term."""
    metrics = state["metrics"]
    app = state["application"]
    
    monthly_payment = app.requested_amount / app.loan_term_months
    monthly_profit = metrics.average_profit / 12
    
    reasoning_parts = []
    is_approved = True
    confidence = 0.8
    
    if monthly_payment > monthly_profit:
        is_approved = False
        reasoning_parts.append(f"Monthly payment {monthly_payment:.2f} exceeds average monthly profit {monthly_profit:.2f}.")
    else:
        reasoning_parts.append(f"Monthly payment {monthly_payment:.2f} is covered by average monthly profit {monthly_profit:.2f}.")
        
    if metrics.current_ratio and metrics.current_ratio < 1.0:
        is_approved = False
        reasoning_parts.append(f"Liquidity is too low (Current Ratio: {metrics.current_ratio:.2f} < 1.0).")
        
    if metrics.debt_to_equity and metrics.debt_to_equity > 2.0:
        is_approved = False
        reasoning_parts.append(f"High leverage detected (Debt to Equity: {metrics.debt_to_equity:.2f} > 2.0).")
        
    reasoning = " | ".join(reasoning_parts)
    if is_approved:
        reasoning += " Standard checks passed."
    else:
        reasoning += " Standard checks failed."
        
    return {"reasoning": reasoning, "is_approved": is_approved, "confidence_score": confidence, "recommended_amount": app.requested_amount if is_approved else 0, "recommended_term": app.loan_term_months if is_approved else 0}


def guardrails_node(state: AgentState):
    """Sanity checks the output of reasoning systems."""
    triggers = []
    
    if not state.get("reasoning"):
        triggers.append("Reasoning was empty.")
        return {"guardrail_passed": False, "guardrail_triggers": triggers, "reasoning": "Guardrail Error: Reasoning was empty."}
        
    if state.get("is_approved") and state.get("confidence_score", 0) < 0.5:
        triggers.append("Low confidence for approval.")
        return {"guardrail_passed": False, "guardrail_triggers": triggers, "is_approved": False, "reasoning": state["reasoning"] + " (Guardrail Overruled: Low confidence for approval)"}

    return {"guardrail_passed": True, "guardrail_triggers": triggers}

def policies_node(state: AgentState):
    """Applies strict bank policies over whatever reasoning decided."""
    metrics = state["metrics"]
    app = state["application"]
    
    policy_violations = []
    is_approved = state["is_approved"]
    
    # Policy 1: Never approve if average profit is strictly negative
    if metrics.average_profit < 0:
        policy_violations.append("Average net profit across years is negative.")
        is_approved = False
        
    # Policy 2: Never approve if loan > 2x fixed assets
    if app.requested_amount > (metrics.fixed_assets_volume * 2):
         policy_violations.append("Loan amount exceeds 200% of collateral (fixed assets).")
         is_approved = False

    return {"is_approved": is_approved, "policy_violations": policy_violations}

def execution_node(state: AgentState):
    """Formats final state into the CreditDecision model."""
    metrics = state["metrics"]
    
    reasoning = state["reasoning"]
    
    decision = CreditDecision(
        is_approved=state["is_approved"],
        recommended_amount=state.get("recommended_amount"),
        recommended_term=state.get("recommended_term"),
        confidence_score=state["confidence_score"],
        reasoning=reasoning,
        policy_violations=state.get("policy_violations", []),
        guardrail_triggers=state.get("guardrail_triggers", []),
        metrics_used=metrics
    )
    
    return {"decision": decision}

# Graph Construction Definition
def route_reasoning(state: AgentState) -> Literal["reasoning_simple", "reasoning_debate", "reasoning_tools"]:
    route = state.get("route", "tools")
    if route == "simple":
        return "reasoning_simple"
    elif route == "debate":
        return "reasoning_debate"
    else:
        return "reasoning_tools"

def build_credit_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("reasoning_simple", reasoning_simple_node)
    workflow.add_node("reasoning_debate", reasoning_debate_node)
    workflow.add_node("reasoning_tools", reasoning_tools_node)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("policies", policies_node)
    workflow.add_node("execution", execution_node)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_reasoning,
        {
            "reasoning_simple": "reasoning_simple",
            "reasoning_debate": "reasoning_debate",
            "reasoning_tools": "reasoning_tools"
        }
    )
    
    # All reasoning nodes go to guardrails
    workflow.add_edge("reasoning_simple", "guardrails")
    workflow.add_edge("reasoning_debate", "guardrails")
    workflow.add_edge("reasoning_tools", "guardrails")
    
    workflow.add_edge("guardrails", "policies")
    workflow.add_edge("policies", "execution")
    workflow.add_edge("execution", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # Test compilation
    app = build_credit_graph()
    print("Graph compiled successfully")
