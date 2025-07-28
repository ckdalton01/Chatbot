# Core imports
import os
from dotenv import load_dotenv
import streamlit as st
import logging

# Streamlit config MUST BE FIRST
st.set_page_config(page_title="DaltonBot3000", page_icon="🐻")

# LangChain imports
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize logging
log_path = os.path.expandvars(r"%PROGRAMDATA%\chatbot")
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, "input_output.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load environment variables
load_dotenv()

# UI Elements
st.title("🐻 DaltonBot3000")

############################### INITIALIZE COMPONENTS #################################

# Embeddings model

# Embeddings model
try:
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text").strip("'\"")
    )
except Exception as e:
    st.error(f"Failed to initialize embeddings: {str(e)}")
    st.stop()

# Chroma vector store
try:
    vector_store = Chroma(
        collection_name=os.getenv("COLLECTION_NAME", "default_collection"),
        embedding_function=embeddings,
        persist_directory=os.getenv("DATABASE_LOCATION", "./chroma_db"),
    )
    # Initialize retriever immediately after vector_store
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    st.error(f"Failed to initialize vector store: {str(e)}")
    st.stop()

# Chat model (DeepSeek R1)
try:
    llm = ChatOllama(
        model="deepseek-r1:8b",
        temperature=0,
        device="cuda",
        num_ctx=8192,
    )
except Exception as e:
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

############################### RAG CHAIN SETUP #######################################

# Prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in a concise, professional manner. If you don't know, say "I don't know."
If possible, include:
Source: source_url
"""
prompt = ChatPromptTemplate.from_template(template)

# Modified RAG chain
def format_chat_history(messages):
    return "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                     for m in messages])

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

############################### CHAT INTERFACE #########################################

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Chat input
user_question = st.chat_input("Ask about Patch My PC documentation...")
if user_question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(content=user_question))
    logging.info(f"USER INPUT: {user_question}")

    # Get AI response
    try:
        ai_response = rag_chain.invoke(
            user_question,  # Pass string directly
            config={"configurable": {"chat_history": format_chat_history(st.session_state.messages)}}
        )
    except Exception as e:
        ai_response = f"Error generating response: {str(e)}"
        logging.error(f"AI ERROR: {str(e)}")


    # Display AI message
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        st.session_state.messages.append(AIMessage(content=ai_response))
    logging.info(f"AI OUTPUT: {ai_response}")