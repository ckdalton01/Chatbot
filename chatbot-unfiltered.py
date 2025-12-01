# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from collections import defaultdict
import datetime
import threading
import getpass
import logging

##############################################
# CMTrace Custom Logging
##############################################

class CMTraceFormatter(logging.Formatter):
    def format(self, record):
        # Time & date
        dt = datetime.datetime.now()
        date = dt.strftime("%m-%d-%Y")
        time = dt.strftime("%H:%M:%S.%f")[:-3]  # milliseconds
        # Timezone bias (minutes)
        tz_bias = int(-datetime.datetime.now().astimezone().utcoffset().total_seconds() / 60)
        # Metadata
        component = "DaltonBot3000"
        context = "System"
        log_type = "1" if record.levelno == logging.INFO else "3"  # 1=Info, 3=Error
        thread_id = threading.get_ident()
        current_user = getpass.getuser()

        # CMTrace message
        log_output = (
            f'<![LOG[{record.getMessage()}]LOG]!>'
            f'<time="{time}+{tz_bias}" '
            f'date="{date}" '
            f'component="{component}" '
            f'context="{context}" '
            f'type="{log_type}" '
            f'thread="{thread_id}" '
            f'file="{current_user}">'
        )
        return log_output


def setup_cmtrace_logger(log_file):
    logger = logging.getLogger("cmtrace")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(CMTraceFormatter())

    logger.addHandler(handler)
    return logger


# Create log directory if it doesn't exist
log_path = os.path.expandvars(r"%PROGRAMDATA%\chatbot")
os.makedirs(log_path, exist_ok=True)

# Setup CMTrace logger
log_file = os.path.join(log_path, "chatbot.log")
logger = setup_cmtrace_logger(log_file)

##############################################
# Load environment variables
##############################################
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

###############################   RETRIEVER TOOL WITH LOGGING   #################################################################################################

@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    logger.info(f"TOOL CALL - Retrieve invoked with query: {query}")

    retrieved_docs = vector_store.similarity_search(query, k=4)

    if not retrieved_docs:
        logger.info("TOOL RESULT - No documents retrieved")
        return "No relevant documents found."

    # Group docs by source
    grouped_docs = defaultdict(list)
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        grouped_docs[source].append(doc.page_content)

    serialized = ""
    for i, (source, contents) in enumerate(grouped_docs.items(), 1):
        preview = contents[0][:200].replace("\n", " ")
        logger.info(f"TOOL RESULT - Source {i}: {source}")
        logger.info(f"TOOL RESULT - Content preview: {preview}...")

        # Merge all chunks for that source into one entry
        combined_content = "\n---\n".join(contents)
        serialized += f"Source: {source}\nContent:\n{combined_content}\n\n"

    return serialized

###############################   AGENT SETUP   #################################################################################################

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

###############################   STREAMLIT APP   #################################################################################################

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
user_question = st.chat_input("You want to know about our Documentation?")

# did the user submit a prompt?
if user_question:
    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    logger.info(f"USER INPUT: {user_question}")

    # invoking the agent
    logger.info("AGENT INVOCATION START")
    result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})
    logger.info("AGENT INVOCATION END")

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        st.session_state.messages.append(AIMessage(ai_message))

    logger.info(f"AI OUTPUT: {ai_message}")
