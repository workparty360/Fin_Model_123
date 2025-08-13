import pandas as pd
from pydantic import BaseModel,Field
from typing import List
from langchain_core.tools import tool
from langchain_core.language_models import BaseLanguageModel

def format_df(df):
    df = df.fillna(0)
    df=df.T
    df=df.fillna(0)
    df.columns=df.iloc[0]
    def normalize_sql_column_names(columns):
        normalized = []
        for i, col in enumerate(columns):
            if pd.isnull(col) or str(col).strip() == '':
                col = f"column_{i}"  
            else:
                col = (
                    str(col)
                    .strip()            
                    .lower()            
                    .replace("+","")
                    .replace("%","")
                    .replace("-"," ")
                    .strip()
                    .replace(" ", "_") 
                    .strip("_") 
                )
            normalized.append(col)
        return normalized
    df.columns = normalize_sql_column_names(df.columns)
    df= df.loc[:,~df.columns.duplicated()].copy()
    df=df.drop(df.index[0])
    return df

class BalanceSheetColumnMap(BaseModel):
    borrowings: List[str] = Field(description="Column for short-term borrowings or long-term-borrowings")
    fixed_assets: List[str] = Field(description="Column for property, plant and equipment")
    investments: List[str] = Field(description="Column for current-investments and non-current-investments")
    cwip: List[str] = Field(description="Column for capital work-in-progress")
    cash_equivalents: List[str] = Field(description="Column for cash and cash equivalents")
    reserves: List[str] = Field(description="Column for retained earnings and reserves")
    equity_capital: List[str] = Field(description="Column for paid-up share capital")

class IncomeStatementColumnMap(BaseModel):
    sales: List[str] = Field(description="Column representing revenue from operations or sales")
    operating_profit: List[str] = Field(description="Column representing earnings before interest and tax or operating profit (e.g., EBIT, EBITDA)")
    net_profit: List[str] = Field(description="Column representing net profit or profit after tax")
    other_income: List[str] = Field(description="Column representing income from non-operating sources like interest or dividends")

class QuaterlyResultColumnMap(BaseModel):
    sales_quaterly: List[str] = Field(description="Column representing revenue from operations or sales")
    operating_profit_quaterly: List[str] = Field(description="Column representing earnings before interest and tax or operating profit (e.g., EBIT, EBITDA)")
    net_profit_quaterly: List[str] = Field(description="Column representing net profit or profit after tax")

def seggregate_into_final_or_not(response):
    queries={}
    final={}
    for name, column in response.items():
        if(len(column)>1):
            queries[name]=column
        else:
            final[name]=column   
    return queries,final

def llm_check_column_formula_validity(
    final_column: str,
    sub_columns: list[str],
    llm = None
) -> str:
    """
    Uses an LLM to check whether sub-columns can be summed to derive the final column.

    Args:
    - final_column: The name of the target column (e.g., 'borrowings')
    - sub_columns: A list of column names to test as parts (e.g., ['long_term_borrowings', 'short_term_borrowings'])

    Returns:
    - "valid", "conditional", or "invalid"
    """

    prompt = f"""
    You are a financial data expert.

    Evaluate whether the following sub-columns can be summed to compute the final financial metric column.

    ### Final Column:
    {final_column}

    ### Sub-Columns:
    {sub_columns}

    ### Instructions:
    - Return ONLY one of the following (no quotes):
    - valid: if the sub-columns clearly sum up to the final column
    - conditional: if it might be correct but depends on context (e.g., naming overlaps)
    - invalid: if summing is clearly wrong or conceptually incorrect

    Do not explain. Just return: valid | conditional | invalid
    """

    if llm is None:
        raise ValueError("You must provide an LLM to run this tool.")

    response = llm.invoke(prompt)
    return response.content.strip().lower()

class ColumnFilterOutput(BaseModel):
    Final_Columns: List[str]

def conditional_columns_rewriter(
    final_column:str,
    sub_columns:list[str],
    parser=None,
    llm= None
):
    format_instructions=parser.get_format_instructions()
    """Used to rewrite subcolumns in form of a list of a given column in financial sheets"""
    prompt = f"""
    You are a financial data expert reviewing balance sheet sub-components.
    
    Your task is to select valid sub-columns that can be used to compute the target financial metric.
    
    Target Metric: {final_column}
    
    Candidate Sub-Columns:
    {sub_columns}
    
    Instructions:
    - Only select sub-columns that are in the same category (Assets, Liabilities, or Equity,Revenue,EBIDTA etc).
    - Do NOT include assets such as "loans and advances" under liabilities like "borrowings".
    - "Borrowings" refers to money the company has taken (liabilities), while "loans and advances" usually refer to money the company has given out (assets).
    - Remove total rows, ambiguous or unrelated columns, and overlaps.
    - Do NOT modify column names.
    - Return only valid sub-columns in their **exact names**.
    
    {format_instructions}
    
    Your output should be a list, e.g.:
    ["string1", "string2"]
    """
    response=llm.invoke(prompt)
    response=parser.parse(response.content)
    return response.Final_Columns
