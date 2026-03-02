from typing import List, Optional
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    """Basic company details."""
    company_name: Optional[str] = Field(None, description="The name of the company")
    tax_id: Optional[str] = Field(None, description="The tax identification number (e.g., INN)")
    industry: Optional[str] = Field(None, description="The industry the company operates in")
    establishment_year: Optional[int] = Field(None, description="The founding year of the company")

class FinancialYearData(BaseModel):
    """Detailed financial data for a specific year."""
    year: int = Field(description="The year the financial data refers to")
    
    # Income Statement
    revenue: Optional[float] = Field(None, description="Total income or sales")
    cogs: Optional[float] = Field(None, description="Cost of goods sold")
    operating_expenses: Optional[float] = Field(None, description="Operating expenses")
    net_profit: float = Field(description="Net profit for the year")
    
    # Balance Sheet
    current_assets: Optional[float] = Field(None, description="Assets expected to be converted to cash within a year")
    fixed_assets: float = Field(description="Volume (accounting value) of fixed assets (Equipment, Real Estate, etc.)")
    current_liabilities: Optional[float] = Field(None, description="Debts due within one year")
    long_term_liabilities: Optional[float] = Field(None, description="Debts due after one year")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities of the company")
    equity: Optional[float] = Field(None, description="Shareholders' equity")
    
    # Cash Flow and Accounts
    operating_cash_flow: float = Field(description="Total operating cash flow for the year")
    investing_cash_flow: Optional[float] = Field(None, description="Cash flow from investing activities")
    financing_cash_flow: Optional[float] = Field(None, description="Cash flow from financing activities")
    cash_and_equivalents: float = Field(description="Average or end-of-year cash balance on accounts")

class ExtractedFinancialData(BaseModel):
    """Data extracted from the raw documents."""
    company_info: Optional[CompanyInfo] = Field(None, description="General metadata about the company")
    yearly_data: List[FinancialYearData] = Field(description="Financial data extracted per year")

class ProcessedMetrics(BaseModel):
    """Calculated averages and sums over the extracted years."""
    average_financial_flow: float = Field(description="Calculated average financial flow over the years")
    average_profit: float = Field(description="Calculated average net profit over the years")
    fixed_assets_volume: float = Field(description="Sum of accounting value of fixed assets (collateral) for the latest year")
    average_cash_balance: float = Field(description="Calculated average money on accounts")
    
    # Ratios (can be None if data is missing)
    current_ratio: Optional[float] = Field(None, description="Current Assets / Current Liabilities for the latest year. Measures liquidity.")
    debt_to_equity: Optional[float] = Field(None, description="Total Liabilities / Total Equity for the latest year.")
    net_profit_margin: Optional[float] = Field(None, description="Average Net Profit / Average Revenue.")
    
class CreditApplication(BaseModel):
    """Input application details from the external system."""
    requested_amount: float = Field(description="The loan amount requested")
    currency: str = Field(default="USD", description="The currency of the requested loan")
    loan_term_months: int = Field(description="The term of the loan in months")
    purpose: Optional[str] = Field(None, description="Purpose of the loan (e.g. Working Capital, Equipment)")
    collateral_proposed: Optional[str] = Field(None, description="Details of the collateral provided by the applicant")

class CreditDecision(BaseModel):
    """Final output to be saved to NoSQL DB."""
    is_approved: bool = Field(description="True if the loan is approved, False otherwise")
    recommended_amount: Optional[float] = Field(None, description="The amount recommended by the agent, may differ from requested")
    recommended_term: Optional[int] = Field(None, description="The loan term in months recommended by the agent")
    confidence_score: float = Field(description="Confidence score of the decision from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed explanation of the reasoning behind the decision")
    policy_violations: List[str] = Field(default_factory=list, description="List of strict bank policies that were violated")
    guardrail_triggers: List[str] = Field(default_factory=list, description="List of guardrails that influenced the decision")
    metrics_used: ProcessedMetrics = Field(description="The metrics that were used to make the decision")
