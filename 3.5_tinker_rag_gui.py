import os
import json
import asyncio
import shutil
import subprocess
from uuid import uuid4
from tkinter import *
from tkinter import ttk, messagebox
from dotenv import load_dotenv, set_key
from firecrawl import AsyncFirecrawlApp, ScrapeOptions
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ENV_PATH = ".env"
load_dotenv(ENV_PATH)

def update_env_var(key, value):
    set_key(ENV_PATH, key, value)
    os.environ[key] = value

def clear_data():
    dataset_path = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
    db_path = os.getenv("DATABASE_LOCATION")
    if os.path.exists(dataset_path): os.remove(dataset_path)
    if os.path.exists(db_path): shutil.rmtree(db_path)
    messagebox.showinfo("Cleared", "data.txt and chroma_db cleared.")

def save_env():
    for key, entry in entries.items():
        update_env_var(key, entry.get())
    messagebox.showinfo("Success", ".env updated")

def run_chatbot():
    subprocess.Popen(["streamlit", "run", "chatbot.py"])

def embed_data():
    dataset_path = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
    if not os.path.exists(dataset_path):
        messagebox.showerror("Error", "data.txt not found")
        return
    chunk_size = int(chunk_size_var.get())
    chunk_overlap = int(chunk_overlap_var.get())
    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    db_path = os.getenv("DATABASE_LOCATION")
    if os.path.exists(db_path): shutil.rmtree(db_path)
    store = Chroma(collection_name=os.getenv("COLLECTION_NAME"), embedding_function=embeddings, persist_directory=db_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    with open(dataset_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    for i, doc in enumerate(docs):
        chunks = splitter.create_documents([doc["raw_text"]], metadatas=[{"source": doc["url"], "title": doc["title"]}])
        ids = [str(uuid4()) for _ in chunks]
        store.add_documents(documents=chunks, ids=ids)
    messagebox.showinfo("Done", f"Embedded {len(docs)} documents.")

def crawl():
    async def run():
        app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        result = await app.crawl_url(
            url=crawl_url.get(),
            limit=int(crawl_limit.get()),
            scrape_options=ScrapeOptions(formats=["markdown"], maxAge=3600000),
            poll_interval=5
        )
        dataset_dir = os.getenv("DATASET_STORAGE_FOLDER")
        os.makedirs(dataset_dir, exist_ok=True)
        output_path = os.path.join(dataset_dir, "data.txt")
        with open(output_path, "a", encoding="utf-8") as f:
            for item in result.data:
                meta = item.metadata
                obj = {
                    "url": meta.get("sourceURL"),
                    "title": meta.get("title"),
                    "raw_text": item.markdown
                }
                f.write(json.dumps(obj) + "\n")
        messagebox.showinfo("Done", f"Crawled {len(result.data)} documents.")
    asyncio.run(run())

# --- UI ---
root = Tk()
root.title("RAG Utility - Tkinter Edition")
root.geometry("700x650")

Label(root, text="Edit Environment Variables", font=("Arial", 14, "bold")).pack(pady=5)
frame = Frame(root)
frame.pack()

entries = {}
env_keys = [
    "EMBEDDING_MODEL", "CHAT_MODEL", "MODEL_PROVIDER", "FIRECRAWL_API_KEY",
    "DATASET_STORAGE_FOLDER", "SNAPSHOT_STORAGE_FILE", "DATABASE_LOCATION", "COLLECTION_NAME"
]

for i, key in enumerate(env_keys):
    Label(frame, text=key).grid(row=i, column=0, sticky="e")
    val = os.getenv(key, "")
    ent = Entry(frame, width=50)
    ent.insert(0, val)
    ent.grid(row=i, column=1, padx=5, pady=2)
    entries[key] = ent

Button(root, text="Save .env", command=save_env).pack(pady=5)

Label(root, text="Crawl Website", font=("Arial", 14, "bold")).pack(pady=5)
crawl_url = Entry(root, width=70)
crawl_url.insert(0, "https://patchmypc.com/kb")
crawl_url.pack()
crawl_limit = Entry(root)
crawl_limit.insert(0, "10")
crawl_limit.pack(pady=2)
Button(root, text="Run Crawl", command=crawl).pack(pady=5)

Label(root, text="Embed Data", font=("Arial", 14, "bold")).pack(pady=5)
chunk_size_var = StringVar(value="1000")
chunk_overlap_var = StringVar(value="200")
Entry(root, textvariable=chunk_size_var).pack()
Entry(root, textvariable=chunk_overlap_var).pack()
Button(root, text="Run Embedding", command=embed_data).pack(pady=5)

Button(root, text="Clear Data & Chroma DB", command=clear_data).pack(pady=10)
Button(root, text="Launch Chatbot (chatbot.py)", command=run_chatbot).pack(pady=10)

root.mainloop()
