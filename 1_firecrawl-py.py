import os
import json
from dotenv import load_dotenv
from firecrawl import AsyncFirecrawlApp, ScrapeOptions

# Load environment variables
load_dotenv()

# Main function
import asyncio

async def crawl_patchmypc_docs():
    app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    print("🚀 Starting crawl of PMPC Docs...")

    crawl_result = await app.crawl_url(
        url="https://docs.patchmypc.com/",
        limit=300,
        scrape_options=ScrapeOptions(
            formats=["markdown"],
            maxAge=3600000  # Use cached results if recent (optional)
        ),
        poll_interval=10  # Firecrawl will wait and poll automatically
    )

    # Optional: inspect raw response
    print("✅ Crawl finished:")
    print(crawl_result.model_dump())

    all_data = crawl_result.data

    # Prepare output
    dataset_dir = os.getenv("DATASET_STORAGE_FOLDER", "dataset/")
    os.makedirs(dataset_dir, exist_ok=True)
    output_file = os.path.join(dataset_dir, "data.txt")

    print(f"💾 Saving {len(all_data)} documents to {output_file}...")



    print("✅ Done! Data saved.")

# Run the async function
if __name__ == "__main__":
    asyncio.run(crawl_patchmypc_docs())
