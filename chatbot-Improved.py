# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain components
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
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

# load .env variables
load_dotenv()

# initialize embedding model
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL")
)

# initialize vector store
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

# initialize LLM
llm = init_chat_model(
    model=os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,
    device="cuda",
    batch_size=8,
)

# prompt template
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. You will be provided with a query for Patch My PC, a software company.
Assume unless otherwise specified, the person is asking about products related to Patch My PC.

Before answering, consider what the underlying intent of the query is, and expand abbreviations or acronyms where needed to improve retrieval.

Use the following retrieved context to answer the user's question.
If you don’t know the answer, say “I don’t know.”

Context:
{context}

User Question:
{question}

Please provide a concise and informative response based on the retrieved information.
If you don't know the answer, say "I don't know" (and don't provide a source).

Return text as follows:
<Answer to the question>
Include the source URL(s) in your response when relevant by citing: "Source: <URL>".
""")

# Streamlit UI setup
st.set_page_config(page_title="DaltonBot3000", page_icon="🐻")
st.title("🐻 DaltonBot3000")

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# replay chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# user input
user_question = st.chat_input("You want to know about Patch My PC Documentation?")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))
    logging.info(f"USER INPUT: {user_question}")

    # similarity search
    retrieved_docs = vector_store.similarity_search(user_question, k=4)

    # concatenate context
    context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    ])

    # format the final prompt
    final_prompt = prompt_template.format(context=context, question=user_question)

    # get LLM response
    ai_response = llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.markdown(ai_response.content)
    st.session_state.messages.append(AIMessage(ai_response.content))
    logging.info(f"AI OUTPUT: {ai_response.content}")
