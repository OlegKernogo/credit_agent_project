import os
import json
from dotenv import load_dotenv

from models import CreditApplication
from scanner import parse_financial_documents, calculate_metrics
from agent import build_credit_graph, AgentState

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

def main():
    # 1. Define the input parameters
    data_folder = "data/"
    print(f"--- 1. Scanning documents in '{data_folder}' ---")
    
    # 2. Extract Data
    extracted_data = parse_financial_documents(data_folder)
    print("\n--- Extracted Raw Data ---")
    print(extracted_data.model_dump_json(indent=2))
    
    # 3. Calculate Metrics
    print("\n--- 2. Calculating Financial Metrics ---")
    metrics = calculate_metrics(extracted_data)
    print(metrics.model_dump_json(indent=2))
    
    # 4. Define Credit Application (from external system)
    # Let's say they want $100,000 for 12 months.
    app = CreditApplication(
        requested_amount=100000.0,
        currency="USD",
        loan_term_months=12,
        purpose="Working Capital to expand operations",
        collateral_proposed="Commercial Real Estate valued at $150k"
    )
    print("\n--- 3. Processing Credit Application ---")
    print(app.model_dump_json(indent=2))
    
    # 5. Run LangGraph Agent
    print("\n--- 4. Running Agentic Pipeline ---")
    graph = build_credit_graph()
    
    initial_state = {
        "metrics": metrics,
        "application": app,
    }
    
    final_state = graph.invoke(initial_state)
    
    # 6. Output Final Decision to JSON (for NoSQL DB)
    print("\n--- 5. Final Credit Decision (JSON Output) ---")
    decision = final_state["decision"]
    print(decision.model_dump_json(indent=2))

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set. Please create a .env file or set it in your environment.")
    else:
        main()
