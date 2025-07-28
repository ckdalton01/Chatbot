import streamlit as st
from dotenv import dotenv_values, set_key
import os
import subprocess

# ==== CONFIG ====
ENV_FILE = ".env"
PROMPT_FILE = "prompt.txt"
CHATBOT_SCRIPTS = ["chatbot.py", "chatbot-improved.py", "chatbotDeepSeek.py"]

# Known models
KNOWN_MODELS = [
    "llama3.2:3b",
    "llama3.3:70b",
]

# ==== Load Values ====
env_vars = dotenv_values(ENV_FILE)
current_model = env_vars.get("CHAT_MODEL", "")
default_script = CHATBOT_SCRIPTS[0]

# Read current prompt
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        current_prompt = f.read()
else:
    current_prompt = ""

# ==== Streamlit UI ====
st.set_page_config(page_title="RAG Config Editor + Launcher", page_icon="🛠️")
st.title("🛠️ RAG Prompt and Script Manager")

# === Script Selection ===
st.subheader("🤖 Choose Chatbot Script")
script_selection = st.selectbox("Select a chatbot script:", CHATBOT_SCRIPTS)

# If chatbotDeepSeek.py is selected, disable other fields
is_limited_mode = script_selection == "chatbotDeepSeek.py"

# === Prompt Editor ===
st.subheader("📄 Prompt Template (prompt.txt)")
if is_limited_mode:
    st.text_area("Prompt Editor (disabled in chatbotDeepSeek.py)", value=current_prompt, height=400, disabled=True)
    st.info("Prompt editing is disabled for chatbotDeepSeek.py — it doesn't use the prompt.")
else:
    prompt_input = st.text_area("Edit your prompt below:", value=current_prompt, height=400)

# === Model Selector ===
st.subheader("🧠 Chat Model (CHAT_MODEL)")
if is_limited_mode:
    st.selectbox("Model selection is disabled in chatbotDeepSeek.py", KNOWN_MODELS, index=KNOWN_MODELS.index(current_model) if current_model in KNOWN_MODELS else 0, disabled=True)
    st.info("Model selection is disabled for chatbotDeepSeek.py — it uses a fixed or different config.")
else:
    model_selection = st.selectbox("Select a chat model:", KNOWN_MODELS, index=KNOWN_MODELS.index(current_model) if current_model in KNOWN_MODELS else 0)

# === Save Button ===
if st.button("💾 Save Changes", disabled=is_limited_mode):
    # Only allow save if not in limited mode
    if not is_limited_mode:
        # Save prompt
        with open(PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(prompt_input)

        # Save model
        set_key(ENV_FILE, "CHAT_MODEL", model_selection)

        st.success("✅ Prompt and model saved!")
    else:
        st.warning("Save disabled in chatbotDeepSeek.py mode.")

# === Launch Script Button ===
if st.button("🚀 Run Selected Chatbot"):
    st.info(f"Launching: {script_selection}")
    subprocess.Popen(["streamlit", "run", script_selection])
    st.success(f"{script_selection} is now running in a new window/terminal.")

# === Exit Button ===
st.subheader("Exit Editor")
if st.button("Exit Editor and Close Streamlit"):
    st.warning("Shutting down Streamlit...")
    st.stop()  # Optional if you want to stop execution here
    sys.exit()

