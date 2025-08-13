from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from typing import Literal,List,Union,Tuple
from langchain.output_parsers import PydanticOutputParser
from datetime import datetime
from pydantic import BaseModel
from data_frame_functions import format_df
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

llm=ChatGroq(model="llama3-70b-8192")
print(llm.invoke("hi"))

class DateListFormat(BaseModel):
    dates:List[Tuple[str, int]]   

parser=PydanticOutputParser(pydantic_object=DateListFormat)
def date_column(df,llm=None,parser=None):
    
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="tool-calling",        
        allow_dangerous_code=True         
    )

    def analyst_node(state: MessagesState):
        """Takes the latest user turn, returns the agent’s answer."""
        user_question = state["messages"][-1].content
        reply = pandas_agent.invoke(user_question)["output"]
        return {
            "messages": state["messages"] + [AIMessage(content=reply)]
        }

    graph_def = StateGraph(MessagesState)       
    graph_def.add_node("analyst", analyst_node) 
    graph_def.set_entry_point("analyst")        
    workflow = graph_def.compile()
    
    i=0
    result_time=None
    response=""
    prompt=""
    for i in range(0,3):
        try:
            conversation = workflow.invoke({
            "messages": [HumanMessage(content=f"""
            From the given DataFrame, identify the column that contains values representing both month and year (such as "Mar 2025", "03/2024", or "2024-12").

            Extract all the values from this column and return them as a list of tuples in the format: (month_abbreviation, year), e.g., [("Mar", 2025), ("Dec", 2024)].

            Rules to follow:
            - If a value contains only the year (e.g., "2025"), assume the month is "Mar" by default.
            - If the month is numeric (e.g., "02" or "2"), convert it to 3-letter month abbreviation ("Feb", etc.).
            - Preserve the original row order.
            - If there are no valid values, return an empty list.

            """
            )]
            })
            response=conversation["messages"][-1].content
            prompt=response+"\nReturn only valid JSON with no extra text.\n"+parser.get_format_instructions()
            result=llm.invoke(prompt)
            result_time=parser.invoke(result.content)
            break
        except:
            pass
    print(response)
    print(prompt)
    if result_time:
        list_dates=result_time.dates
        df.index=list_dates
        df.index=[datetime.strptime(f"{month} {year}", "%b %Y") for month, year in df.index]
    else:
        raise NotImplementedError("Parser Could Not Parse")

    return df

df=pd.read_excel(r"C:\Users\HP\Documents\Balance_Sheet_Concor.xlsx")
df=format_df(df=df)
df=date_column(df=df,llm=llm,parser=parser)
print(df)
