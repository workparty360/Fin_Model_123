import streamlit as st
from data_frame_functions import format_df,BalanceSheetColumnMap
from data_frame_functions import IncomeStatementColumnMap,QuaterlyResultColumnMap
from data_frame_functions import seggregate_into_final_or_not,llm_check_column_formula_validity
from data_frame_functions import ColumnFilterOutput,conditional_columns_rewriter
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel,Field
from typing import List,Literal,Union,Sequence,Annotated,TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from itertools import zip_longest
import json
import requests
from bs4 import BeautifulSoup
import time
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
import operator
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()
st.title("Finance Bot")
st.write("Ensure that the Balance sheet and Income statements are in the same time frame")
st.write("It is necessary to upload all 3 docs")
st.write("Suggested to use screener to create the docs")
st.header("Upload File")
file_type = st.selectbox("Select file type", ["Excel","CSV"])
balance_sheet = st.file_uploader("Choose an Excel Balance Sheet file", type=["xlsx", "xls","csv"])
income_statement = st.file_uploader("Choose an Excel Income Statement Sheet file", type=["xlsx", "xls","csv"])
quaterly_report = st.file_uploader("Choose an Excel Quaterly Report file", type=["xlsx", "xls","csv"])

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
if "balance_sheet" not in st.session_state:
    st.session_state.balance_sheet=pd.DataFrame()
if "income_statement" not in st.session_state:
    st.session_state.income_statement=pd.DataFrame()
if "quater" not in st.session_state:
    st.session_state.quater=pd.DataFrame()

if file_type=="CSV":    
    st.session_state.income_statement=pd.read_csv(income_statement)
    st.session_state.balance_sheet=pd.read_csv(balance_sheet)
    st.session_state.quater=pd.read_csv(quaterly_report)


if file_type=="Excel":    
    st.session_state.income_statement=pd.read_excel(income_statement)
    st.session_state.balance_sheet=pd.read_excel(balance_sheet)
    st.session_state.quater=pd.read_excel(quaterly_report)

# st.session_state.income_statement=pd.read_excel(r"C:\Users\HP\Documents\Mahindra_income_statements.xlsx")
# st.session_state.balance_sheet=pd.read_excel(r"C:\Users\HP\Documents\Mahindra Balance Sheets.xlsx")
# st.session_state.quater=pd.read_excel(r"C:\Users\HP\Documents\Mahindra_Quaterly_Reports.xlsx")



st.session_state.balance_sheet=format_df(st.session_state.balance_sheet)
st.session_state.income_statement=format_df(st.session_state.income_statement)
st.session_state.quater=format_df(st.session_state.quater)

if "start_row" not in st.session_state:
    st.session_state.start_row = 1979
if "end_row" not in st.session_state:
    st.session_state.end_row  = 1979

st.session_state.start_row = st.slider("Select 1st Column Year", min_value=1979, max_value=2025, value=1979)
if st.session_state.start_row == 1979:
    st.warning("Please select a year greater than 1979.")
    st.button("Submit START", disabled=True)
    st.stop()
else:
    if st.button("Submit START"):
        st.success(f"Data submitted for start year {st.session_state.start_row}")

st.session_state.end_row=st.slider("Select Last Column Year", min_value=1979, max_value=2025, value=1979)
if st.session_state.end_row == 1979:
    st.warning("Please select a year greater than 1979.")
    st.button("Submit END", disabled=True)
    st.stop()
else:
    if st.button("Submit END"):
        st.success(f"Data submitted for end year {st.session_state.end_row}")

start=st.session_state.start_row
end=st.session_state.end_row
if "time" not in st.session_state:
    st.session_state.time=[]

if st.session_state.start_row>st.session_state.end_row:
    st.session_state.time=[i for i in range(st.session_state.end_row,st.session_state.start_row+1)]
    st.session_state.time=st.session_state.time[::-1]
elif st.session_state.start_row<st.session_state.end_row:
    st.session_state.time=[i for i in range(st.session_state.start_row,st.session_state.end_row+1)]

st.session_state.balance_sheet.index=st.session_state.time
st.session_state.income_statement.index=st.session_state.time
st.session_state.quater.index = pd.to_datetime(st.session_state.quater.index, format="%b '%y")
    
print(st.session_state.time)

st.session_state.balance_sheet = st.session_state.balance_sheet.sort_index(ascending=True)
st.session_state.income_statement=st.session_state.income_statement.sort_index(ascending=True)
st.session_state.quater=st.session_state.quater.sort_index(ascending=True)

conn=sqlite3.connect("my_database.db")
st.session_state.balance_sheet.to_sql(con=conn,name="balance_sheet",if_exists="replace")
st.session_state.income_statement.to_sql(con=conn,name="income_statement",if_exists="replace")
st.session_state.quater.to_sql(con=conn,name="quaterly_report",if_exists="replace")
conn.commit()

db=SQLDatabase.from_uri("sqlite:///my_database.db")
toolkit=SQLDatabaseToolkit(llm=llm, db=db)
tools=toolkit.get_tools()

schema_balance_sheet=tools[1].invoke("balance_sheet")
schema_income_statement=tools[1].invoke("income_statement")
schema_quaterly_report=tools[1].invoke("quaterly_report")

parser_balance_sheet = PydanticOutputParser(pydantic_object=BalanceSheetColumnMap)
prompt_balance_sheet = PromptTemplate(
    template="""
    You are a finance data expert.
    
    Your task is to find the best column names from the provided list that match these financial concepts:
    - borrowings
    - fixed_assets
    - investments
    - cwip
    - cash_equivalents
    - reserves
    - equity_capital
    
    ### Column Choices:
    {column_choices}
    
    ### Instructions:
    - First identify the section of each column as either: Asset, Liability, or Equity.
    - For each concept, return a matching column name (same as one provided in the list) from the same section.
    - If no suitable column is found, return an empty list: []
    - Do not match borrowings with any liability or provisions
    - Only use column names from the provided list.
    - Output only a valid Pydantic object that matches the following schema:
    
    {format_instructions}
    """
    ,
    input_variables=["column_choices"],
    partial_variables={"format_instructions": parser_balance_sheet.get_format_instructions()},
)

parser_income_statement = PydanticOutputParser(pydantic_object=IncomeStatementColumnMap)
prompt_income_statement = PromptTemplate(
    template="""
You are a finance data expert.

Your task is to find the best column names from the provided list that match these key **income statement** concepts:
- sales
- operating_profit
- net_profit
- other_income

### Column Choices:
{column_choices}

### Instructions:
- Use your domain knowledge to identify correct mappings for income statement terms.
- For each concept, return AT MOST one matching column name, exactly as it appears in the list.
- Only use column names from the provided list.
- If no suitable match is found, return an empty list: []
- Do NOT infer or modify column names; return only those that exactly match from the input.
- Output must be a valid pydantic object that matches the schema:

{format_instructions}
""",
    input_variables=["column_choices"],
    partial_variables={"format_instructions": parser_income_statement.get_format_instructions()},
)

parser_quaterly_report=PydanticOutputParser(pydantic_object=QuaterlyResultColumnMap)
prompt_quaterly_report=PromptTemplate(
    template="""
You are a finance data expert.

Your task is to find the best column names from the provided list that match these key **income statement** concepts:
- sales
- operating_profit
- net_profit

### Column Choices:
{column_choices}

### Instructions:
- Use your domain knowledge to identify correct mappings for income statement terms.
- For each concept, return AT MOST one matching column name, exactly as it appears in the list.
- Only use column names from the provided list.
- If no suitable match is found, return an empty list: []
- Do NOT infer or modify column names; return only those that exactly match from the input.
- Output must be a valid pydantic object that matches the schema:

{format_instructions}
""",
    input_variables=["column_choices"],
    partial_variables={"format_instructions": parser_quaterly_report.get_format_instructions()},
)
print(st.session_state.balance_sheet)
print(st.session_state.income_statement)
print(st.session_state.quater)

response_balance_sheet=llm.invoke(prompt_balance_sheet.invoke({"column_choices":schema_balance_sheet}))
response_income_statement=llm.invoke(prompt_income_statement.invoke({"column_choices":schema_income_statement}))
response_quaterly_report=llm.invoke(prompt_quaterly_report.invoke({"column_choices":schema_quaterly_report}))


response_balance_sheet=dict(parser_balance_sheet.parse(response_balance_sheet.content))
response_income_statement=dict(parser_income_statement.parse(response_income_statement.content))
response_quaterly_report=dict(parser_quaterly_report.parse(response_quaterly_report.content))


queries_balance_sheet,final_balance_sheet=seggregate_into_final_or_not(response=response_balance_sheet)
queries_income_statement,final_income_statement=seggregate_into_final_or_not(response=response_income_statement)
queries_quaterly_report,final_quaterly_report=seggregate_into_final_or_not(response=response_quaterly_report)

columns_balance_sheet=list(queries_balance_sheet.keys())
sub_columns_balance_sheet=list(queries_balance_sheet.values())

columns_income_statement=list(queries_income_statement.keys())
sub_columns_income_statement=list(queries_income_statement.values())

columns_quaterly_report=list(queries_quaterly_report.keys())
sub_columns_quaterly_report=list(queries_quaterly_report.values())

parser_columnmap=PydanticOutputParser(pydantic_object=ColumnFilterOutput)

for i,column in enumerate(columns_balance_sheet):
    response=llm_check_column_formula_validity(final_column=column,sub_columns=sub_columns_balance_sheet[i],llm=llm)
    if(response=="valid"):
        queries_balance_sheet[column]=sub_columns_balance_sheet[i]
    if(response=="conditional"):
        queries_balance_sheet[column]=conditional_columns_rewriter(final_column=column,sub_columns=sub_columns_balance_sheet[i],llm=llm,parser=parser_columnmap)
    if(response=="invalid"):
        continue

for i,column in enumerate(columns_income_statement):
    response=llm_check_column_formula_validity(final_column=column,sub_columns=sub_columns_income_statement[i],llm=llm)
    if(response=="valid"):
        queries_income_statement[column]=sub_columns_income_statement[i]
    if(response=="conditional"):
        queries_income_statement[column]=conditional_columns_rewriter(final_column=column,sub_columns=sub_columns_income_statement[i],llm=llm,parser=parser_columnmap)
    if(response=="invalid"):
        continue

for i,column in enumerate(columns_quaterly_report):
    response=llm_check_column_formula_validity(final_column=column,sub_columns=sub_columns_quaterly_report[i],llm=llm)
    if(response=="valid"):
        queries_quaterly_report[column]=sub_columns_quaterly_report[i]
    if(response=="conditional"):
        queries_quaterly_report[column]=conditional_columns_rewriter(final_column=column,sub_columns=sub_columns_quaterly_report[i],llm=llm,parser=parser_columnmap)
    if(response=="invalid"):
        continue

f={}
q={}

length=len(st.session_state.balance_sheet.index)

for name,list_names in final_balance_sheet.items():
    try:
        f[name]=list(st.session_state.balance_sheet.loc[:,list_names[0]])
    except:
        f[name]=[0 for i in range(0,length)]
        continue

for name,list_names in final_income_statement.items():
    try:
        f[name]=list(st.session_state.income_statement.loc[:,list_names[0]])
    except:
        f[name]=[0 for i in range(0,length)]
        continue

for name,list_names in final_quaterly_report.items():
    try:
        q[name]=list(st.session_state.quater.loc[:,list_names[0]])
    except:
        q[name]=[0 for i in range(0,length)]
        continue


def sum_lists_elementwise(list_of_lists):
    """
    Sums multiple lists element-wise, padding with 0 for unequal lengths.
    
    Args:
        list_of_lists (List[List[float]]): A list of numerical lists.
        
    Returns:
        List[float]: A single list with the element-wise sums.
    """
    return [sum(tup) for tup in zip_longest(*list_of_lists, fillvalue=0)]


for name,list_names in queries_balance_sheet.items():
    list_common=[]
    for i,sub_name in enumerate(list_names):
        try:
            list_common.append(list(st.session_state.balance_sheet.loc[:,sub_name]))
        except:
            continue
    f[name]=sum_lists_elementwise(list_common)

for name,list_names in queries_income_statement.items():
    list_common=[]
    for i,sub_name in enumerate(list_names):
        try:
            list_common.append(list(st.session_state.income_statement.loc[:,sub_name]))
        except:
            continue
    f[name]=sum_lists_elementwise(list_common)

for name,list_names in queries_quaterly_report.items():
    list_common=[]
    for i,sub_name in enumerate(list_names):
        try:
            list_common.append(list(st.session_state.quater.loc[:,sub_name]))
        except:
            continue
    q[name]=sum_lists_elementwise(list_common)

par=pd.DataFrame(f)
qat=pd.DataFrame(q)

sales=list(par.loc[:,'sales'])
operating_profit=list(par.loc[:,'operating_profit'])
net_profit=list(par.loc[:,'net_profit'])
other_income=list(par.loc[:,'other_income'])
cash_equivalents=list(par.loc[:,'cash_equivalents'])
investments=list(par.loc[:,'investments'])
equity_capital=list(par.loc[:,'equity_capital'])
reserves=list(par.loc[:,'reserves'])
borrowings=list(par.loc[:,'borrowings'])
fixed_assets=list(par.loc[:,'fixed_assets'])
cwip=list(par.loc[:,'cwip'])
sales_quaterly=list(qat.loc[:,'sales_quaterly'])
operating_profit_quaterly=list(qat.loc[:,'operating_profit_quaterly'])
net_profit_quaterly=list(qat.loc[:,'net_profit_quaterly'])

yearly_growth={}
prof_ratio={}
debt_risk={}
yoy_growth={}
quaterly_growth={}

prof_ratio["other_income_as_percent_of_net_profit"]=[(float(other)/net_profit[i]*100) for i,other in enumerate(other_income)]
prof_ratio["other_income_as_percent_of_investments"]=[other/(cash_equivalents[i]+investments[i])*100 for i,other in enumerate(other_income)]
prof_ratio["operating_profit_margin_yearly"]=[(other/sales[i]*100) for i,other in enumerate(operating_profit)]
prof_ratio["net_profit_margin_yearly"]=[(other/sales[i]*100) for i,other in enumerate(net_profit)]
prof_ratio["return_on_equity"]=[net/(equity_capital[i]+reserves[i])*100 for i,net in enumerate(net_profit)]
prof_ratio["return_on_capital_employed"]=[(net/(equity_capital[i]+reserves[i]+borrowings[i])*100) for i,net in enumerate(operating_profit)]
debt_risk["share_holders_funds"]=[net+equity_capital[i] for i,net in enumerate(reserves)]
debt_risk["fixed_assets_sum_cwip"]=[fixed+cwip[i] for i,fixed in enumerate(fixed_assets)]
debt_risk["debt_to_equity"]=[borrowings[i]/shares for i,shares in enumerate(debt_risk["share_holders_funds"])]
yearly_growth["sales_yearly_growth"]=[(sales[i+1]-sale)/sale*100 for i,sale in enumerate(sales) if i+1<len(sales)]    
yearly_growth["operating_profit_yearly_growth"]=[(operating_profit[i+1]-ope)/ope*100 for i,ope in enumerate(operating_profit) if i+1<len(operating_profit)]
yearly_growth["net_profit_yearly_growth"]=[(net_profit[i+1]-net)/net*100 for i,net in enumerate(net_profit) if i+1<len(net_profit)]
yoy_growth["sales_yoy_growth"]=[(sales_quaterly[i+4]-sale)/sale*100 for i,sale in enumerate(sales_quaterly) if i+4<len(sales_quaterly)]
yoy_growth["operating_profit_yoy_growth"]=[((operating_profit_quaterly[i+4]-ope)/ope)*100 for i,ope in enumerate(operating_profit_quaterly) if i+4<len(operating_profit_quaterly)]
yoy_growth["net_profit_yoy_growth"]=[((net_profit_quaterly[i+4]-net)/net)*100 for i,net in enumerate(net_profit_quaterly) if i+4<len(net_profit_quaterly)]
quaterly_growth["operating_profit_margin"]=[operating_profit_quaterly[i]/sale*100 for i,sale in enumerate(sales_quaterly)]
quaterly_growth["net_profit_margin"]=[(net_profit_quaterly[i]/sale*100) for i,sale in enumerate(sales_quaterly)]

yearly_growth_df=pd.DataFrame(yearly_growth)
yoy_growth_df=pd.DataFrame(yoy_growth)
debt_risk_df=pd.DataFrame(debt_risk)
prof_ratio_df=pd.DataFrame(prof_ratio)
quaterly_growth_df=pd.DataFrame(quaterly_growth)

x=list(st.session_state.balance_sheet.index)
y=sales
plt.plot(x,y)
plt.title("Sales")
plt.xlabel("Time") 
plt.ylabel("Sales")
plt.grid(True)
plt.show()
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Sine Wave")
st.pyplot(fig)

yearly_growth_df.index=list(st.session_state.balance_sheet.index)[1:]
yoy_growth_df.index=list(st.session_state.quater.index)[4:]
debt_risk_df.index=list(st.session_state.balance_sheet.index)
prof_ratio_df.index=list(st.session_state.balance_sheet.index)
quaterly_growth_df.index=list(st.session_state.quater.index)
if "st.session_state.dfs" not in st.session_state:
    st.session_state.dfs=""
st.session_state.dfs = {
    "Yearly_Growth": json.loads(yearly_growth_df.to_json(orient="index")),
    "Profitability_Ratios": json.loads(prof_ratio_df.to_json(orient="index")),
    "Quaterly_Growth": json.loads(quaterly_growth_df.to_json(orient="index")),
    "Year On Year Growth": json.loads(yoy_growth_df.to_json(orient="index")),
    "Debt Risk": json.loads(debt_risk_df.to_json(orient="index")),
}

if "response_data" not in st.session_state:
    st.session_state.response_data=""


st.session_state.response_data=llm.invoke(f"Analyse the data and submit 5 points on each type highlight risks with red flags and positives with pos but do not highlight every point\n {st.session_state.dfs}")
st.write('\nYearlyGrowth\n')
st.dataframe(yearly_growth_df)
st.write("\n Yoy growth\n")
st.dataframe(yoy_growth_df)
st.write('\nDebt Risk\n')
st.dataframe(debt_risk_df)
st.write('\nProf Ratio\n')
st.dataframe(prof_ratio_df)
st.write('\nQuaterly Growth\n')
st.dataframe(quaterly_growth_df)

st.write(f"**{st.session_state.response_data.content}**")

if "ticker" not in st.session_state:
    st.session_state.ticker=""

st.session_state.ticker=st.text_input("TICKER :")

if st.session_state.ticker=="":
    st.stop()
else:
    pass

st.session_state.ticker=st.session_state.ticker.strip().upper()
values=[]
labels = [
    "Current Price",
    "Day Range",
    "52 Week Range",
    "Market Cap",
    "Volume",
    "P/E Ratio",
    "Dividend Yield",
    "Exchange",
    "1:","2:","3:"
]
company_summary=""
try: 
    url=f"https://www.google.com/finance/quote/{st.session_state.ticker}:NSE"
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    price = soup.find('div', {'data-last-price': True})['data-last-price']
    exchange = soup.find('div', {'data-exchange': True})['data-exchange']
    articles = soup.find_all('div', class_='yY3Lee')
    divs = soup.find_all("div", class_="P6K39c")

    for i, div in enumerate(divs):
        values.append(div.text.strip())

    company_summary = "\n".join(f"{label}: {value}" for label, value in zip(labels, values))
    st.write(company_summary)

    price_1=""
    price_1 +=f"st.session_state.ticker: {st.session_state.ticker}"
    price_1 +=f"Price: {price}"
    print(f"st.session_state.ticker: {st.session_state.ticker}")
    print(f"Price: {price}")
    print(exchange)
    all_articles_text = ""

    for article in articles:
        source = article.find('div', class_='sfyJob')
        time = article.find('div', class_='Adak')
        title = article.find('div', class_='Yfwt5')
        link = article.find('a', href=True)['href']

        article_text = f"Source: {source.text if source else 'N/A'}\n"
        article_text += f"Time: {time.text if time else 'N/A'}\n"
        article_text += f"Title: {title.text if title else 'N/A'}\n"
        article_text += f"Link: {link}\n\n"

        all_articles_text +=("News "+article_text)
    st.write(f"**{all_articles_text}**")
    if "company_summary" not in st.session_state:
        st.session_state.company_summary=company_summary
except:
    pass

balance_sheet=st.session_state.balance_sheet.to_json(orient='index')
income_statements=st.session_state.income_statement.to_json(orient='index')
quaterly_reports=st.session_state.quater.to_json(orient='index')

string1="BALANCE SHEET TO FIND OTHER INCOME,BORROWINGS,FIXED ASSETS ETC:"+str(balance_sheet)+"\nINCOME STATEMENT TO FIND OPERATING PROFIT,NET PROFIT,SALES"+str(income_statement)+"\nQUATERLY REPORT TO FIND THE QUATERLY SALES, PROFIT"+str(quaterly_reports)

string2=str(st.session_state.dfs)
llm=ChatGroq(model="gemma2-9b-it")
docs=[Document(page_content=string1),
    Document(page_content=string2),
    Document(page_content=all_articles_text),
    Document(page_content=price_1),
    Document(page_content=st.session_state.response_data.content),
    Document(page_content=st.session_state.company_summary)]

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=70)
result=splitter.split_documents(docs)
vector_str=Chroma.from_documents(documents=result,collection_name="sample",embedding=embedding_model)

retriever = vector_str.as_retriever(search_kwargs={'k':5})

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_financial_data",
    """Search information from balance_sheets,
    income_statements,quaterly_reports or from market news, price"""
)

tools = [retriever_tool]
retrieve=ToolNode([retriever_tool])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def ai_assistant(state:AgentState):
    print("---CALL AGENT---")
    messages = state['messages']
    
    if len(messages)>1:
        last_message = messages[-1]
        question = last_message.content
        prompt=PromptTemplate(
        template="""You are a helpful financial assistant whatever question has been asked to find out that in the given question and answer.
                        Here is the question:{question}
                        Share:{ticker}
                        """,
                        input_variables=["question","ticker"]
                        )
            
        chain = prompt | llm
    
        response=chain.invoke({"question": question,"ticker":st.session_state.ticker})
        print(response.content)
        return {"messages": [response]}
    else:
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        print(response.content)
        return {"messages": [response]}
    
class grade(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")

def grade_documents(state:AgentState)->Literal["Output_Generator", "Query_Rewriter"]:
    llm_with_structure_op=llm.with_structured_output(grade)
    
    prompt=PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
                    Here is the document: {context}
                    Here is the user’s question: {question}
                    If the document talks about or contains information related to the user’s question, mark it as relevant. 
                    Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
                    input_variables=["context", "question"]
                    )
    chain = prompt | llm_with_structure_op
    
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generator" 
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewriter" 
    
    
def generate(state:AgentState):
    print("---GENERATE---")
    messages = state["messages"]

    question = messages[0].content
    
    last_message = messages[-1]
    docs = last_message.content
    
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = prompt | llm

    response = rag_chain.invoke({"context": docs, "question": question})
    print(f"AI:{response}")
    
    return {"messages": [response]}

def rewrite(state:AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning. 
                    Here is the initial question: {question} 
                    Formulate an improved question: """)
       ]
    response = llm.invoke(message)
    return {"messages": [response]}

memory = MemorySaver()

workflow=StateGraph(AgentState)
workflow.add_node("My_Ai_Assistant",ai_assistant)
workflow.add_node("Vector_Retriever", retrieve) 
workflow.add_node("Output_Generator", generate)
workflow.add_node("Query_Rewriter", rewrite) 
workflow.add_edge(START,"My_Ai_Assistant")
workflow.add_conditional_edges("My_Ai_Assistant",
                            tools_condition,
                            {"tools": "Vector_Retriever",
                                END: END,})
workflow.add_conditional_edges("Vector_Retriever",
                            grade_documents,
                            {"generator": "Output_Generator",
                            "rewriter": "Query_Rewriter"
                            }
                            )
workflow.add_edge("Output_Generator", END)
workflow.add_edge("Query_Rewriter", "My_Ai_Assistant")
app=workflow.compile(checkpointer=memory)
if "history" not in st.session_state:
    st.session_state.history=[]
user_input = st.text_input("Enter your Question:")
st.session_state.history.append("Human: "+user_input)
response=app.invoke({"messages":[HumanMessage(content=user_input)]},config={"configurable":{"thread_id":"abcd1234"}})
st.write(f"AI:{response.get("messages")[-1].content}")
st.session_state.history.append(f"AI:{response.get("messages")[-1].content}")
st.write(st.session_state.history)
for i in (st.session_state.history):
    print(i)

