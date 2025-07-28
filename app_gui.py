import os
import asyncio
from uuid import uuid4
from dotenv import load_dotenv, set_key
import streamlit as st
from firecrawl import AsyncFirecrawlApp, ScrapeOptions
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import json

# Paths
ENV_PATH = ".env"
load_dotenv(ENV_PATH)

# Utility to update .env
def update_env(key, value):
    set_key(ENV_PATH, key, value)
    os.environ[key] = value

# Crawl function
async def crawl_url(url, limit, formats, max_age, output_file):
    app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    result = await app.crawl_url(
        url=url,
        limit=limit,
        scrape_options=ScrapeOptions(formats=formats, maxAge=max_age),
        poll_interval=5
    )
    data = result.data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for item in data:
            d = item.model_dump()
            meta = d.get("metadata", {})
            obj = {"url": meta.get("sourceURL"), "title": meta.get("title"), "raw_text": d.get("markdown")}
            f.write(json.dumps(obj) + "\n")
    return len(data)

# Embed function
def embed_data(dataset_file, chunk_size, chunk_overlap):
    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    db_path = os.getenv("DATABASE_LOCATION")
    if os.path.exists(db_path): shutil.rmtree(db_path)
    store = Chroma(collection_name=os.getenv("COLLECTION_NAME"), embedding_function=embeddings, persist_directory=db_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    with open(dataset_file, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    count = 0
    for entry in lines:
        docs = splitter.create_documents([entry['raw_text']], metadatas=[{"source": entry['url'], "title": entry['title']}])
        ids = [str(uuid4()) for _ in docs]
        store.add_documents(documents=docs, ids=ids)
        count += len(docs)
    return count

# Utility to clear data and DB
def clear_data(dataset_file, db_path):
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

# Streamlit UI
st.title("RAG Dashboard")
# .env fields
st.header("Configure .env")
with st.form("env_form"):
    embed_model = st.text_input("Embedding Model", value=os.getenv("EMBEDDING_MODEL"))
    chat_model = st.text_input("Chat Model", value=os.getenv("CHAT_MODEL"))
    provider = st.text_input("Model Provider", value=os.getenv("MODEL_PROVIDER"))
    api_key = st.text_input("Firecrawl API Key", value=os.getenv("FIRECRAWL_API_KEY"))
    data_folder = st.text_input("Dataset Folder", value=os.getenv("DATASET_STORAGE_FOLDER"))
    snapshot_file = st.text_input("Snapshot File", value=os.getenv("SNAPSHOT_STORAGE_FILE"))
    db_loc = st.text_input("Chroma DB Location", value=os.getenv("DATABASE_LOCATION"))
    coll = st.text_input("Chroma Collection", value=os.getenv("COLLECTION_NAME"))
    if st.form_submit_button("Save .env"):
        for key, val in [
            ("EMBEDDING_MODEL", embed_model), ("CHAT_MODEL", chat_model),
            ("MODEL_PROVIDER", provider), ("FIRECRAWL_API_KEY", api_key),
            ("DATASET_STORAGE_FOLDER", data_folder), ("SNAPSHOT_STORAGE_FILE", snapshot_file),
            ("DATABASE_LOCATION", db_loc), ("COLLECTION_NAME", coll)
        ]:
            update_env(key, val)
        st.success(".env updated!")

# Crawl section
st.header("Crawl Website")
with st.form("crawl_form"):
    target_url = st.text_input("URL to Scrape", "https://patchmypc.com/kb")
    limit = st.number_input("Document Limit", min_value=1, value=10)
    formats = st.multiselect("Formats", options=["markdown", "html"], default=["markdown"])
    max_age = st.number_input("Max Cache Age (ms)", min_value=0, value=3600000)
    if st.form_submit_button("Run Crawl"):
        out_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
        count = asyncio.run(crawl_url(target_url, limit, formats, max_age, out_file))
        st.success(f"Crawled {count} documents.")

# Embed section
st.header("Chunk & Embed Data")
with st.form("embed_form"):
    chunk_size = st.number_input("Chunk Size", min_value=100, value=1000)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=200)
    if st.form_submit_button("Run Embedding"):
        dataset_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
        docs = embed_data(dataset_file, chunk_size, chunk_overlap)
        st.success(f"Embedded {docs} document chunks.")

# Clear data and DB
st.header("Clear Data & DB")
if st.button("Clear All Scraped Data and Database"):
    data_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
    db_path = os.getenv("DATABASE_LOCATION")
    clear_data(data_file, db_path)
    st.warning("data.txt and chroma_db cleared.")

# Chatbot launcher
st.header("Launch Chatbot")
if st.button("Run Chatbot"):
    st.info("Please run: streamlit run chatbot.py in your terminal.")
