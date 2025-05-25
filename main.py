import openai
import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import re
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document




def extract_numbers(text):
  """Extracts all numbers (integers and decimals) from a string.

  Args:
    text: The input string.

  Returns:
    A list of numbers found in the string.
  """
  return [float(s) for s in re.findall(r'-?\d+\.?\d*', text)]

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

import pandas as pd 
df = pd.read_csv("/Users/shikhashah/Desktop/Projects/HR_management/Resume.csv")

df.head()

llm = ChatOpenAI(temperature = 0.5)

agent_executer = create_csv_agent(llm, '/Users/shikhashah/Desktop/Projects/HR_management/Resume.csv', verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

agent_output = agent_executer.invoke("How many HR have equal to or more than 15 years of experience are there? Give the ID of the HR.")

#print (agent_output)

numbers = extract_numbers(agent_output["output"])

int_numbers = [int(x) for x in numbers]

print (int_numbers)

df1 = pd.DataFrame()

#print (df[df['ID'] == int_numbers[0]])

for i in range (len(int_numbers)):
	df2 = df[df['ID'] == int_numbers[i]]
	
	column_text = df2["Resume_str"].dropna().astype(str).str.cat(sep=",")
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
	docs = text_splitter.create_documents([column_text])
	summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
	summary = summary_chain.run(docs)
	print("Summary:\n", summary)


	#print (df2)
	#df = pd.concat([df, df2], ignore_index=True)
	#print (df)

#df = df.drop(index=df.index[:len(int_numbers)])
#df = df.reset_index()

#print (df)