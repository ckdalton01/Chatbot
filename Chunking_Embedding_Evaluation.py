import os
import json
import shutil
import gc
import time
from uuid import uuid4
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

###################################################################################################
# 1. ENV + MODELS
###################################################################################################

load_dotenv()

DATASET_FOLDER = os.getenv("DATASET_STORAGE_FOLDER")
CLEAN_FILE = os.path.join(DATASET_FOLDER, "dataKB_Blogs.txt")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

###################################################################################################
# 2. DATA LOADER
###################################################################################################

def load_clean_dataset():
    """Load pre-cleaned dataset (from Script 1)."""
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

###################################################################################################
# 3. VECTORSTORE BUILDER
###################################################################################################

def build_vectorstore(docs, chunk_size, chunk_overlap, persist_dir):
    """Build Chroma vectorstore with given chunk settings."""
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    vector_store = Chroma(
        collection_name=f"docs_{chunk_size}_{chunk_overlap}",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    for doc in docs:
        chunks = splitter.create_documents(
            [doc["raw_text"]],
            metadatas=[{"source": doc["url"], "title": doc["title"]}]
        )
        ids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(chunks, ids=ids)

    return vector_store

###################################################################################################
# 4. EVALUATION
###################################################################################################

def evaluate_config(vector_store, test_queries, k=3):
    """Check retrieval metrics for each query."""
    hits_at_1 = 0
    hits_at_k = 0
    mrr_total = 0

    results = []

    for q in test_queries:
        query = q["question"]
        expected_url = q["expected_url"]

        docs = vector_store.similarity_search(query, k=k)
        retrieved_urls = [d.metadata["source"] for d in docs]

        # Hit@1
        if retrieved_urls and retrieved_urls[0] == expected_url:
            hits_at_1 += 1

        # Hit@k
        if expected_url in retrieved_urls:
            hits_at_k += 1
            rank = retrieved_urls.index(expected_url) + 1
            mrr_total += 1 / rank

        results.append((query, expected_url, retrieved_urls))

    num_queries = len(test_queries)
    metrics = {
        "Hit@1": hits_at_1 / num_queries,
        "Hit@k": hits_at_k / num_queries,
        "MRR": mrr_total / num_queries,
    }
    return metrics, results

###################################################################################################
# 5. MAIN
###################################################################################################

if __name__ == "__main__":
    docs = load_clean_dataset()
    print(f"Loaded {len(docs)} cleaned documents.")

    # Define configs to test
    configs = [
        (400, 80),
        (600, 120),
        (800, 200),
        (1000, 300),
        (1200, 350),
        (1400, 400),
        (1600, 500),
        (1800, 600),
    ]

    # Define your evaluation queries (ground truth mapping)
    test_queries = [
        {"question": "How do I build a custom app?",
         "expected_url": "https://docs.patchmypc.com/patch-my-pc-cloud/custom-apps/create-a-custom-app"},
        {"question": "How do I install the Patch My PC Publisher for Configuration Manager?",
         "expected_url": "https://docs.patchmypc.com/installation-guides/configmgr/download-and-run-the-msi"},
        {"question": "What is dual scan?",
         "expected_url": "https://patchmypc.com/blog/sccm-co-management-dual-scan/"},
        {"question": "How do I migrate from On-prem publisher to Cloud portal",
         "expected_url": "https://patchmypc.com/kb/migrating-from-publisher-to-cloud/"},
        {"question": "How do I install Advanced Insights?",
         "expected_url": "https://docs.patchmypc.com/patch-my-pc-insights/download-and-install-insights"},
        {"question": "ADR is failing with 0X87D20417",
         "expected_url": "https://patchmypc.com/kb/adr-error-0x87d20417/"},
        {"question": "I changed a setting in the Publisher, should I republish my app?",
         "expected_url": "https://patchmypc.com/kb/when-how-republish-patch-my/"},


        # Add more as needed
    ]

    results_summary = []
    created_dirs = []

    for size, overlap in configs:
        persist_dir = f"chroma_db_{size}_{overlap}"
        created_dirs.append(persist_dir)

        print(f"\n=== Testing config: chunk_size={size}, overlap={overlap} ===")
        vs = build_vectorstore(docs, size, overlap, persist_dir=persist_dir)

        # --- Run evaluation ---
        metrics, details = evaluate_config(vs, test_queries, k=3)
        results_summary.append((size, overlap, metrics))

        # Print detailed results per query
        for q, expected, retrieved in details:
            hit = expected in retrieved
            print(f"Q: {q}\n  Expected: {expected}\n  Hit: {hit}\n  Retrieved: {retrieved}\n")

        # Print metrics summary for this config
        print(f"Metrics: Hit@1={metrics['Hit@1']:.2%}, Hit@3={metrics['Hit@k']:.2%}, MRR={metrics['MRR']:.3f}")

        # --- Cleanup this vectorstore safely ---
        vs = None
        gc.collect()
        time.sleep(5)  # allow file handles to close
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                print(f"✅ Removed {persist_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove {persist_dir}: {e}")

    print("\n=== FINAL SUMMARY ===")
    for size, overlap, metrics in results_summary:
        print(f"Config (size={size}, overlap={overlap}) -> "
              f"Hit@1={metrics['Hit@1']:.2%}, Hit@3={metrics['Hit@k']:.2%}, MRR={metrics['MRR']:.3f}")

    print("\n✅ All temporary test databases cleaned up after evaluation.")