import os
import json
import asyncio
import time
from dotenv import load_dotenv
from firecrawl import AsyncFirecrawlApp, ScrapeOptions
from firecrawl.types import CrawlJobResponse, CrawlStatusResponse

# Load environment variables
load_dotenv()

# Timeout polling function for checking crawl status
async def wait_for_crawl_completion(app, crawl_id, timeout=300, interval=5):
    """
    Waits for a Firecrawl crawl job to complete with a timeout.

    Args:
        app: AsyncFirecrawlApp instance
        crawl_id: ID of the crawl job
        timeout: Max time in seconds to wait
        interval: Polling interval in seconds

    Returns:
        List of scraped page data
    """
    start_time = time.time()
    attempt = 0

    while True:
        status = await app.check_crawl_status(crawl_id)
        state = status.status

        print(f"⏱️  Attempt {attempt}: status = {state} ({status.completed} / {status.total})")
        if state == "completed":
            return status.data

        if time.time() - start_time > timeout:
            raise TimeoutError(f"🔥 Timeout after {timeout} seconds waiting for crawl to complete.")

        attempt += 1
        await asyncio.sleep(interval)

# Main async crawl function
async def crawl_patchmypc_docs():
    app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    print("🚀 Starting crawl of https://docs.patchmypc.com...")
    crawl_result = await app.crawl_url(
        url="https://docs.patchmypc.com",
        limit=50,  # Adjust as needed
        scrape_options=ScrapeOptions(formats=["markdown"])
    )

    print("🔎 Raw crawl result:")
    print(crawl_result.model_dump())

    # Handle fast vs delayed crawls
    if isinstance(crawl_result, CrawlStatusResponse):
        print("✅ Crawl already completed.")
        all_data = crawl_result.data
    else:
        print("⏳ Crawl returned a job ID. Polling until complete...")
        crawl_id = crawl_result.id
        all_data = await wait_for_crawl_completion(app, crawl_id)

    # Save to file
    dataset_dir = os.getenv("DATASET_STORAGE_FOLDER", "dataset/")
    os.makedirs(dataset_dir, exist_ok=True)
    output_file = os.path.join(dataset_dir, "data.txt")

    print(f"💾 Saving {len(all_data)} pages to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            json_obj = {
                "url": item.get("url") or item.get("metadata", {}).get("sourceURL", ""),
                "title": item.get("metadata", {}).get("title", ""),
                "raw_text": item.get("markdown", "")
            }
            f.write(json.dumps(json_obj) + "\n")

    print(f"✅ Done! All data written to {output_file}")

# Entry point
if __name__ == "__main__":
    asyncio.run(crawl_patchmypc_docs())
