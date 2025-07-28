import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

input_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
output_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "dataKB_Blogs.txt")

# Initialize LLM
chat = ChatOllama(model=os.getenv("CHAT_MODEL"))

def generate_title(text: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant. Your task is to extract 3-4 concise keywords that describe this document. When formatting the output do not add numbers, stars or anything other than the 3-4 words separated by commas."),
        HumanMessage(content=f"Document text:\n{text}\n\nReturn only the keywords, no explanation.")
    ]
    try:
        response = chat.invoke(messages)
        title = response.content.strip().replace('"', '')
        print(f"Generated Title: {title}")
        return title
    except Exception as e:
        print(f"Error generating title: {e}")
        return ""

##Low quality removal
def is_low_quality(text: str) -> bool:
    return "login" in text.lower() or "copyright" in text.lower() or len(text.split()) < 10

seen_texts = set()

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        try:
            data = json.loads(line)
            raw_text = data.get("raw_text", "")
            if raw_text in seen_texts:
                print("Duplicate raw_text found, skipping.")
                continue
            seen_texts.add(raw_text)

            if not data.get("title") and len(raw_text) > 30:
                title = generate_title(raw_text[:2500])
                data["title"] = title

            outfile.write(json.dumps(data) + "\n")
        except json.JSONDecodeError:
            print("Skipping invalid JSON line.")
