import os
import json
import shutil
from uuid import uuid4
from dotenv import dotenv_values, set_key, load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import subprocess

# Load environment
load_dotenv()
env_file = ".env"
prompt_file = "prompt.txt"
db_path = os.getenv("DATABASE_LOCATION")
dataset_path = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "dataKB_Blogs.txt")
script_options = ["chatbot.py", "chatbot-improved.py", "chatbotdeepseek.py"]
known_models = [
    "llama3.2:3b", "llama3.3:70b"
]

env_vars = dotenv_values(env_file)
current_model = env_vars.get("CHAT_MODEL", known_models[0])
current_prompt = ""
if os.path.exists(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        current_prompt = f.read()

st.set_page_config(page_title="RAG Dashboard")
st.title("RAG System Dashboard")

# Script selection
st.subheader("Select Chatbot Script")
selected_script = st.selectbox("Chatbot script to run:", script_options)
is_deepseek = selected_script == "chatbotdeepseek.py"

# Prompt editor
st.subheader("Edit Prompt (prompt.txt)")
if is_deepseek:
    st.text_area("Prompt content (disabled)", current_prompt, height=300, disabled=True)
    st.info("Prompt editing is disabled when using chatbotdeepseek.py.")
else:
    prompt_input = st.text_area("Prompt content", value=current_prompt, height=300)

# Model selector
st.subheader("Select LLM Model (CHAT_MODEL)")
if is_deepseek:
    st.selectbox("Model (disabled)", known_models, index=known_models.index(current_model), disabled=True)
else:
    selected_model = st.selectbox("Model:", known_models, index=known_models.index(current_model))

# Save button
if st.button("Save Prompt and Model", disabled=is_deepseek):
    if not is_deepseek:
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_input)
        set_key(env_file, "CHAT_MODEL", selected_model)
        st.success("Prompt and model saved.")

# Delete chroma DB
st.subheader("Vector Store Management")
if st.button("Delete Vector DB"):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        st.success("chroma_db deleted.")
    else:
        st.info("No chroma_db directory found.")

# Chunking controls
st.subheader("Chunking Parameters")
chunk_size = st.slider("Chunk size", min_value=128, max_value=2048, step=64, value=1600)
chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=1024, step=32, value=500)

# Embedding
st.subheader("Embed Data into Vector Store")

embedding_disabled = os.path.exists(db_path)

if embedding_disabled:
    st.info("Embedding already completed. To re-embed, delete the existing chroma_db first.")

with st.container():
    embed_btn = st.button("Start Embedding", disabled=embedding_disabled)

    if embed_btn and not embedding_disabled:
        st.info("Starting embedding process...")
        log_area = st.container()
        log_placeholder = log_area.empty()

        embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        vector_store = Chroma(
            collection_name=os.getenv("COLLECTION_NAME"),
            embedding_function=embeddings,
            persist_directory=db_path,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        with open(dataset_path, encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        log_lines = []
        for i, line in enumerate(lines):
            url = line.get("url", "unknown")
            raw_text = line.get("raw_text", "")
            title = line.get("title", "Untitled")

            if not raw_text.strip():
                continue

            docs = splitter.create_documents([raw_text], metadatas=[{"source": url, "title": title}])
            ids = [str(uuid4()) for _ in range(len(docs))]
            vector_store.add_documents(docs, ids=ids)

            log_line = f"[{i+1}/{len(lines)}] Embedded: {url}"
            log_lines.append(log_line)
            log_placeholder.text("\n".join(log_lines[-15:]))  # Display last 15 entries only

        st.success("Embedding complete.")


# Launch chatbot script
st.subheader("Run Chatbot")
if st.button("Run Selected Chatbot"):
    st.info(f"Launching: {selected_script}")
    subprocess.Popen(["streamlit", "run", selected_script])
    st.success(f"{selected_script} is running in a new terminal window.")

# Exit Streamlit app
st.subheader("Exit Dashboard")
if st.button("Exit and Close Streamlit"):
    st.warning("Shutting down...")
    st.stop()
    os._exit(0)
