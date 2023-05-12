from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import AzureChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import openai
import string
import ast
import sqlite3
from datetime import timedelta
import os
import pandas as pd
import numpy as np
import random
from urllib import parse
import re
import json
from sqlalchemy import create_engine  
import sqlalchemy as sql
from plotly.graph_objects import Figure as PlotlyFigure
from matplotlib.figure import Figure as MatplotFigure
import time
from langchain.utilities import PythonREPL
import sys
from io import StringIO

template = """Your are a data engineer in a data science team. Your team is given a task by business to work on. You have a team leader and a data scientist colleages. 
You know source systems very well and is tasked to prepare the dataset according to the instruction from the team lead. The data scientist has skills in data science to find insights from the dataset prepared by you.
You have access to the following tools:

{tools}
You  analyze the ask to understand what is required. Once you understand the data that is required, you will retrieve the database schema and write query inside python code 
to retrieve data and perform any additional cleaning and transformation step. Finally, you persist result data for use by your team. 
You are given following  functions to use in your python code help you retrieve data and persist your result for use by your data scientist.
    1. execute_sql_query(sql_query: str): A Python function can query data from the database given the query. 
        - To use this function that you need to create a sql query which has to be syntactically correct for SQLITE. 
        - execute_sql_query returns a Python pandas dataframe contain the results of the query.
    2. persist(df: Pandas, name:str): this function help you persist pandas dataframe to a storage for later use
    3. load(name): this function is to load a previously persisted dataset. It returns a pandas dataframe
wrap your python code inside ```python ```

Use the following format:

Ask: description of the ask you must perform 
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to stay within the scope of your ask. Do not do the job of the data scientist

Ask: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
def execute_sql_query(query, limit=10000):  
    db_path = "../data/northwind.db"
    engine = create_engine(f'sqlite:///{db_path}')  
    result = pd.read_sql_query(query, engine)
    result = result.infer_objects()
    for col in result.columns:  
        if 'date' in col.lower():  
            result[col] = pd.to_datetime(result[col], errors="ignore")  

    if limit is not None:  
        result = result.head(limit)  # limit to save memory  

    return result  
def get_table_schema(not_used):
  
  
    # Define the SQL query to retrieve table and column information 
    sql_query = """    
    SELECT m.name AS TABLE_NAME, p.name AS COLUMN_NAME, p.type AS DATA_TYPE  
    FROM sqlite_master AS m  
    JOIN pragma_table_info(m.name) AS p  
    WHERE m.type = 'table'  
    """  
    # Execute the SQL query and store the results in a DataFrame  
    df = execute_sql_query(sql_query, limit=None)  
    output=[]
    # Initialize variables to store table and column information  
    current_table = ''  
    columns = []  
    
    # Loop through the query results and output the table and column information  
    for index, row in df.iterrows():
        table_name = f"{row['TABLE_NAME']}" 

        column_name = row['COLUMN_NAME']  
        data_type = row['DATA_TYPE']   
        if " " in table_name:
            table_name= f"[{table_name}]" 
        column_name = row['COLUMN_NAME']  
        if " " in column_name:
            column_name= f"[{column_name}]" 

        # If the table name has changed, output the previous table's information  
        if current_table != table_name and current_table != '':  
            output.append(f"table: {current_table}, columns: {', '.join(columns)}")  
            columns = []  
        
        # Add the current column information to the list of columns for the current table  
        columns.append(f"{column_name} {data_type}")  
        
        # Update the current table name  
        current_table = table_name  
    
    # Output the last table's information  
    output.append(f"table: {current_table}, columns: {', '.join(columns)}")
    output = "\n ".join(output)
    return output

def run_python(python_code):

    
    python_code=python_code.strip().strip("```python")
    """Run command and returns anything printed."""
    # sys.stderr.write("EXECUTING PYTHON CODE:\n---\n" + command + "\n---\n")
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        exec(python_code, globals())
        sys.stdout = old_stdout
        output = mystdout.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        output = "encountered error: "+str(e)
    # sys.stderr.write("PYTHON OUTPUT: \"" + output + "\"\n")
    if len(output)==0:
        output = "python program run successfully without any output. If you want to observe output, you need to use print"
    return output
def persist(df, name):
    df.to_parquet(name)
    # print(f"persist df under {name}")
    return f"persist the input data under {name}"
def load(name):
    return pd.read_parquet(name)
