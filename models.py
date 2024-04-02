import subprocess

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
