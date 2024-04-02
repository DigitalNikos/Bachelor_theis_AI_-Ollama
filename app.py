import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import Rag
from models import get_ollama_models


# Set the page configuration for Streamlit
st.set_page_config(page_title="💬 AI Chatbot")

def initialize_or_update_assistant(model_name):
    # Initializes or updates the global assistant object based on the selected model
    if "assistant" in st.session_state:
        st.session_state["assistant"].update_model(model_name=model_name)
    else:
        st.session_state["assistant"] = Rag(model_name=model_name)

def on_model_selection_change():
    # Callback for handling changes in model selection from the sidebar
    model_name = st.session_state.model_selection
    initialize_or_update_assistant(model_name)

def display_messages():
    # Displays user and assistant messages in the chat
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    # Processes the user's message and generates a response from the assistant
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_text, True))

        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
        
        st.session_state["assistant"].conversation_history.extend([user_text, agent_text])
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    # Reads uploaded files, saves temporarily, and ingests content through the assistant
    st.session_state["assistant"].clear()

    sidebar_placeholder = st.session_state.get("sidebar_status_placeholder")
    
    for file in st.session_state["file_uploader"]:
        _, file_extension = os.path.splitext(file.name)
        file_extension = file_extension.lower()
        print(f"FILE: {file}")
        # if validate_file(st, file):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        sidebar_placeholder.markdown(f"🔄 Ingesting {file.name}...")

        st.session_state["assistant"].ingest(file_path, file_extension)
        os.remove(file_path)

        sidebar_placeholder.markdown("✅ Ingestion complete!")

def process_url_input():
    # Ingests content from the provided URL through the assistant
    url = st.session_state["url_input"]
    if url and len(url.strip()) > 0:
        st.session_state["assistant"].clear()
        
        sidebar_placeholder = st.session_state.get("sidebar_status_placeholder")
        sidebar_placeholder.markdown(f"🔄 Ingesting content from {url}...")
        st.session_state["assistant"].ingest_from_url(url)
        sidebar_placeholder.markdown("✅ Ingestion complete!")
        
        st.session_state["url_input"] = ""

def page():
    # Main function to structure the Streamlit page
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # st.session_state["assistant"] = Rag(model_name=selected_model.lower())
        st.session_state["url_input"] = ""

    st.header("💬 AI Chatbot")

    if "model_selection" not in st.session_state:
        st.session_state["model_selection"] = "mistral"  # Default model selection
    
    model_options = ["mistral", "llama2"] 
    st.sidebar.selectbox("Select LLM Model :robot_face::", model_options, key="model_selection", on_change=on_model_selection_change)
    
    if "assistant" not in st.session_state:
        initialize_or_update_assistant(st.session_state.model_selection)

    global file_details_placeholder
    file_details_placeholder = st.sidebar.empty()
    st.sidebar.text_input("Enter a URL", key="url_input", on_change=process_url_input)
    st.sidebar.file_uploader("Upload document", type=["pdf", "txt", "doc", "docx"], key="file_uploader", accept_multiple_files=True, on_change=read_and_save_file)

    if "sidebar_status_placeholder" not in st.session_state:
        st.session_state["sidebar_status_placeholder"] = st.sidebar.empty()

    display_messages()
    st.chat_input("Message", key="user_input", on_submit=process_input)

if __name__ == "__main__":
    print(f"Print models: {get_ollama_models()}")
    page()
