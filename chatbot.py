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
import re

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
    """Initialize logger with CMTrace-compatible formatting."""
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

BRAND_NAME = os.getenv("BRAND_NAME", "Patchy")
BRAND_ICON = os.getenv("BRAND_ICON", "🐻")
BRAND_DOCS_NAME = os.getenv("BRAND_DOCS_NAME", "Patch My PC Documentation")

##############################################
# Initialize embeddings model
##############################################
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

##############################################
# Initialize Chroma vector store
##############################################
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

##############################################
# Initialize chat model
##############################################
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,
    # GPU optimization
    device="cuda",
    batch_size=8,
    # CPU/memory optimization
    n_threads=24,
    context_window=8192,
    # Performance optimizations
    quantization="int8",
    use_flash_attention=True
)

##############################################
# Load system prompt template
##############################################
prompt_file_path = os.getenv("PROMPT_FILE")
with open(prompt_file_path, "r", encoding="utf-8") as f:
    prompt_template_str = f.read()

prompt_template_str = prompt_template_str.replace("{BRAND_DOCS_NAME}", BRAND_DOCS_NAME)
prompt = PromptTemplate.from_template(prompt_template_str)
##############################################
# Retriever tool with logging
##############################################
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    logger.info(f"TOOL CALL - Retrieve invoked with query: {query}")

    retrieved_docs = vector_store.similarity_search(query, k=6)

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
        logger.info(f"TOOL RESULT - Source {i}: {source}")

        # Merge all chunks for that source into one entry
        combined_content = "\n---\n".join(contents)
        serialized += f"Source: {source}\nContent:\n{combined_content}\n\n"

    return serialized

##############################################
# Response relevance validation (0–100 scale)
##############################################
def validate_relevance(user_question: str, ai_message: str, threshold: int = 60) -> bool:
    """
    Validate if the response is relevant to the query using a 0–100 score.
    Returns True if the score >= threshold, False otherwise.
    """
    validation_prompt = f"""
    You are a strict relevance evaluator. 
    You will compare a user question and an assistant response.

    User question: "{user_question}"
    Assistant response: "{ai_message}"

    Rules:
    1. Focus only on the user’s question. If the assistant discusses competitor products and not {BRAND_DOCS_NAME}, score very low (0–20).
    2. If the assistant answer is mostly about the right product/topic but phrased differently (e.g., describing features instead of answering directly), it is still relevant. Give a high score (70–100).
    3. If the assistant answer mixes relevant information with unrelated or noisy content, treat it as "partially relevant." Give a medium score (40–69).
    4. If the answer is completely off-topic, give a very low score (0–20).
    
    
    Output format:
    SCORE: <0–100>
    JUSTIFICATION: <short explanation why you chose this score>
    """

    validation_result = llm.invoke([HumanMessage(content=validation_prompt)])
    raw_output = validation_result.content.strip()

    # Log raw validation result
    logger.info(f"VALIDATION RAW OUTPUT: {raw_output}")

    # Extract score
    match = re.search(r"(\d{1,3})", raw_output)
    score = int(match.group(1)) if match else 0

    logger.info(f"VALIDATION INPUT - Question: {user_question}")
    logger.info(f"VALIDATION INPUT - Response: {ai_message}")
    logger.info(f"VALIDATION SCORE: {score}")

    return score >= threshold

##############################################
# Agent setup
##############################################
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

##############################################
# Streamlit app #DaltonBot3000
##############################################
st.set_page_config(page_title=BRAND_NAME, page_icon=BRAND_ICON)
st.title(f"{BRAND_ICON} {BRAND_NAME}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_question = st.chat_input(f"You want to know about {BRAND_DOCS_NAME}?")

if user_question:
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    logger.info(f"USER INPUT: {user_question}")

    # Invoke the agent
    logger.info("AGENT INVOCATION START")
    result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})
    logger.info("AGENT INVOCATION END")

    ai_message = result["output"]

    # Validate relevance with scoring
    is_relevant = validate_relevance(user_question, ai_message, threshold=60)

    if is_relevant:
        # Show validated AI response
        with st.chat_message("assistant"):
            st.markdown(ai_message)
            st.session_state.messages.append(AIMessage(ai_message))
        logger.info(f"AI OUTPUT ACCEPTED: {ai_message}")
    else:
        # Handle irrelevant AI response
        with st.chat_message("assistant"):
            st.markdown(f"I'm afraid I do not have the answers you are looking for. I'm trained on {BRAND_DOCS_NAME} only.")
        logger.info("AI OUTPUT REJECTED - Response below threshold")
