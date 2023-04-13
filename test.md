from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import os
os.environ["OPENAI_API_KEY"] = "7123602c74c8447e9137b562388ae55c"

#Note: The openai-python library support for Azure OpenAI is in preview.
import openai
openai.api_type = "azure"
openai.api_base = "https://pmopenai.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-dv3",
  prompt="",
  temperature=1,
  max_tokens=100,
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None)

llm = AzureOpenAI(deployment_name="text-dv3", model_name="text-dv3")
text_splitter = CharacterTextSplitter()

with open("./mystory.txt") as f:
    mystory = f.read()
texts = text_splitter.split_text(mystory)
print(texts)



docs = [Document(page_content=t) for t in texts[:3]]



prompt_template = """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN CHINESE:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
chain.run(docs)