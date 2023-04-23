from langchain.chat_models import AzureChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
import os
import openai
import math
import faiss
#from GenerativeAgentClass import GenerativeAgent

os.environ["OPENAI_API_KEY"] = "7123602c74c8447e9137b562388ae55c"
os.environ["OPENAI_API_BASE"] = "https://pmopenai.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")


#response = openai.ChatCompletion.create(
  #engine="turbo-35",
  #messages = [{"role":"system","content":""}],
  #temperature=1,
  #max_tokens=400,
  #top_p=0.95,
  #frequency_penalty=0,
  #presence_penalty=0,
  #stop=None)
        
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)    

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


USER_NAME = "Person A" 

LLM = AzureChatOpenAI(deployment_name="turbo-35", model_name="gpt-3.5-turbo")

tommie = GenerativeAgent(name="Tommie", 
              age=25,
              traits="anxious, likes design", # You can add more persistent traits here 
              status="looking for a job", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                   "Drove across state to move to a new town but doesn't have a job yet."
               ],
               reflection_threshold = 8, # we will give this a relatively low number to show how reflection works
             )

# We can give the character memories directly
tommie_memories = [
    "Tommie remembers his dog, Bruno, from when he was a kid",
    "Tommie feels tired from driving so far",
    "Tommie sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Tommie is hungry",
    "Tommie tries to get some rest.",
]
for memory in tommie_memories:
    tommie.add_memory(memory)

#print(interview_agent(tommie, "can you briefly describe yourself?"))

def chatbot_response(human_input):
    return interview_agent(tommie, human_input)
