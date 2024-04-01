import os
from urllib.parse import urlparse

def file_type(file):
    _, ext = os.path.splitext(file.name)
    return ext.lower()

# Function to validate the uploaded file
def validate_file(st, uploaded_file):
    if uploaded_file is not None:
        ext = file_type(uploaded_file)

        st.sidebar.write("Uploaded File Details:")
        st.sidebar.write("Filename:", uploaded_file.name)
        st.sidebar.write("File Type:", ext)
        st.sidebar.write("File Size:", uploaded_file.size, "bytes")

        if ext in ['.pdf', '.docx', '.txt']:
            st.sidebar.success("File format is supported!")
            return True
        else:
            st.sidebar.error("File format not supported. Please upload a PDF, DOCX, or TXT file.")
            return False
    else:
        st.sidebar.write("Please upload a file.")
        return False

# Function to validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Function to check file type
def file_type(file):
    # Use os.path.splitext to extract file extension
    _, ext = os.path.splitext(file.name)
    return ext.lower()