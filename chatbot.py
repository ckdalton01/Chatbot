# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import logging

# Create log directory if it doesn't exist
log_path = os.path.expandvars(r"%PROGRAMDATA%\chatbot")
os.makedirs(log_path, exist_ok=True)

# Configure logging
log_file = os.path.join(log_path, "input_output.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# load environment variables
load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

###############################   INITIALIZE CHAT MODEL   #######################################################################################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,
    # GPU optimization
    device="cuda",  # Force GPU usage
    batch_size=8,  # Good for RTX 4070's 12GB VRAM
    # CPU/memory optimization
    n_threads=24,  # 24 physical cores of your i9-14900HX
    context_window=8192,  # Leverage your 96GB RAM
    # Performance optimizations
    quantization="int8",  # Better performance with minimal quality loss
    use_flash_attention=True  # If supported
)

# pulling prompt from hub
prompt_file_path = os.getenv("PROMPT_FILE")

with open(prompt_file_path, "r", encoding="utf-8") as f:
    prompt_template_str = f.read()

prompt = PromptTemplate.from_template(prompt_template_str)


# creating the retriever tool
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)

    serialized = ""

    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"

    return serialized


# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="DaltonBot3000", page_icon="🐻")
st.title("🐻 DaltonBot3000")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
user_question = st.chat_input("You want to know about Patch My PC Documentation?")

# did the user submit a prompt?
if user_question:
    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))
    logging.info(f"USER INPUT: {user_question}")  # <-- log input

    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))
    logging.info(f"AI OUTPUT: {ai_message}")  # <-- log output
