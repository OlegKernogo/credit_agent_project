import os
from dotenv import load_dotenv
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from models import ExtractedFinancialData, ProcessedMetrics, FinancialYearData
import pdfplumber

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a TXT or PDF file."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def parse_financial_documents(folder_path: str) -> ExtractedFinancialData:
    """Reads all documents in a folder and extracts financial data using an LLM."""
    all_text = ""
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and (filepath.endswith('.txt') or filepath.endswith('.pdf') or filepath.endswith('.md')):
            try:
                all_text += f"\n--- Document Core: {filename} ---\n"
                all_text += extract_text_from_file(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # load_dotenv()  # Load environment variables for API keys            
    # Use LLM to extract structured data
    if os.getenv("USE_OPENROUTER") == "true":
        llm = ChatOpenAI(
            model="openai/gpt-5.2",    
            openai_api_key = os.getenv("OPENROUTER_API_KEY"),
            openai_api_base = 'https://openrouter.ai/api/v1',
            # streaming=True,
            temperature=0
        )
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0) # You will need to set OPENAI_API_KEY env var

  
    
    structured_llm = llm.with_structured_output(ExtractedFinancialData)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract yearly financial data (Income Statement, Balance Sheet) from the provided documents. Extract data for all available years (at least 2)."),
        ("user", "Here are the documents:\n\n{text}")
    ])
    
    chain = prompt | structured_llm
    
    result: ExtractedFinancialData = chain.invoke({"text": all_text})
    return result

def calculate_metrics(extracted_data: ExtractedFinancialData) -> ProcessedMetrics:
    """Calculates averages and required metrics from the extracted yearly data."""
    years_data = extracted_data.yearly_data
    num_years = len(years_data)
    
    if num_years == 0:
        raise ValueError("No yearly data extracted to calculate metrics.")
        
    total_financial_flow = sum([y.operating_cash_flow for y in years_data])
    total_profit = sum([y.net_profit for y in years_data])
    total_cash = sum([y.cash_and_equivalents for y in years_data])
    
    # We take the most recent fixed assets volume as collateral representation
    # Sort by year to ensure we get the latest
    sorted_years = sorted(years_data, key=lambda x: x.year, reverse=True)
    latest_year_data = sorted_years[0]
    latest_fixed_assets = latest_year_data.fixed_assets
    
    # Calculate new optional ratios based on the latest year or averages
    current_ratio = None
    if latest_year_data.current_assets and latest_year_data.current_liabilities:
        current_ratio = latest_year_data.current_assets / latest_year_data.current_liabilities
        
    debt_to_equity = None
    if latest_year_data.total_liabilities and latest_year_data.equity:
        debt_to_equity = latest_year_data.total_liabilities / latest_year_data.equity
        
    net_profit_margin = None
    average_revenue = sum([y.revenue for y in years_data if y.revenue]) / len([y.revenue for y in years_data if y.revenue]) if any(y.revenue for y in years_data) else 0
    average_profit = total_profit / num_years
    if average_revenue > 0:
        net_profit_margin = average_profit / average_revenue
    
    metrics = ProcessedMetrics(
        average_financial_flow=(total_financial_flow / num_years),
        average_profit=average_profit,
        average_cash_balance=(total_cash / num_years),
        fixed_assets_volume=latest_fixed_assets,
        current_ratio=current_ratio,
        debt_to_equity=debt_to_equity,
        net_profit_margin=net_profit_margin
    )
    return metrics

if __name__ == "__main__":
    # Test script will go here
    pass
