import os
import json
from dotenv import load_dotenv
from scraper import PoliteWebScraper

load_dotenv()

scraper = PoliteWebScraper("https://docs.patchmypc.com/")
scraped_data = scraper.scrape_site(max_pages=500)

# Format and write to dataset folder
dataset_dir = os.getenv("DATASET_STORAGE_FOLDER")
os.makedirs(dataset_dir, exist_ok=True)

output_file = os.path.join(dataset_dir, "data3.txt")

# Extensions to skip
skip_extensions = (
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.tiff',
    '.mp3', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.mkv', '.webm',
    '.zip', '.rar', '.tar', '.gz', '.7z', '.exe', '.bin', '.dll', '.msi',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.js', '.css', '.woff', '.woff2', '.ttf', '.otf'
)

skip_patterns = [
    "/wp-content/uploads/",
    "/assets/",
    "/static/"
]
with open(output_file, "a", encoding="utf-8") as f:
    for url, content in scraped_data.items():
        lower_url = url.lower()
        if lower_url.endswith(skip_extensions) or any(p in lower_url for p in skip_patterns):
            continue
        print(url)
        json_obj = {
            "url": url,
            "title": "",  # Optionally extract title from scraper
            "raw_text": content
        }
        f.write(json.dumps(json_obj) + "\n")

print(f"Scraped and saved {len(scraped_data)} pages to {output_file}")