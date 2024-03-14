from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import pandas as pd
from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

openai_client = OpenAI()

df = pd.read_csv(r"data\cleaned_grievance_v2.csv")

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

result = agent.run("which department has the more grievances?")

print(result)