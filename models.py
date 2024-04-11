import subprocess
from rag import Rag

def get_ollama_models():
    """
    Executes the 'ollama list' command and parses the output to list available model names, excluding versions and additional details.

    Returns:
    - A list of strings, where each string is a model name.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        models = [line.split('\t')[0].split(':')[0] for line in lines if line.strip() and not line.startswith('NAME')]
        return models
    except subprocess.CalledProcessError as e:
        print("Failed to list ollama models:", e)
        return []
    
def initialize_or_update_assistant(st, model_name):
    # Initializes or updates the global assistant object based on the selected model
    if "assistant" in st.session_state:
        st.session_state["assistant"].update_model(model_name=model_name)
    else:
        st.session_state["assistant"] = Rag(model_name=model_name)

def on_model_selection_change(st):
    # Callback for handling changes in model selection from the sidebar
    model_name = st.session_state.model_selection
    print(f"model_name: {model_name}")
    initialize_or_update_assistant(model_name)
