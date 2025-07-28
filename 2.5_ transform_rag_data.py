import json
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET_STORAGE_FOLDER', 'dataset/')


def clean_text(text: str) -> str:
    """Remove boilerplate and normalize text for your specific content"""
    unwanted_phrases = [
        "Share this:", "Like Loading...", "Leave a comment",
        "Subscribe Subscribed", "Privacy & Cookies:", "Cookie Policy",
        "Loading Comments...", "Write a Comment...", "Design a site like this"
    ]
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
    return text.strip()


def transform_data(input_path: str, output_path: str):
    """Transform data to match your existing pipeline requirements"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            try:
                original_entry = json.loads(line.strip())
                cleaned_text = clean_text(original_entry.get("raw_text", ""))

                # Create the exact format your pipeline expects
                transformed_entry = {
                    "url": original_entry.get("url", ""),
                    "title": original_entry.get("title", ""),
                    "raw_text": cleaned_text  # Using cleaned text but preserving structure
                }

                outfile.write(json.dumps(transformed_entry) + "\n")

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")


if __name__ == "__main__":
    # Ensure dataset folder exists
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    input_filename = os.path.join(DATASET_FOLDER, "data.txt")
    output_filename = os.path.join(DATASET_FOLDER, "data2.txt")

    print(f"Transforming {input_filename} -> {output_filename}...")
    transform_data(input_filename, output_filename)
    print(f"Transformation complete. Optimized data saved to {output_filename}")