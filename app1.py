import streamlit as st
import os
import tempfile
# Make sure chatbot_core.py is in the same directory
from chatbot_core import load_documents, build_qa_chain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Document Chatbot (OpenAI)", layout="wide")
# Updated Title to include XLSX
st.title("💬 Chat with your Documents (PDF, XPT, SAS7BDAT, XLSX) using OpenAI")
st.caption("Powered by LangChain, OpenAI, and Streamlit")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None


# --- Helper Function to Save Uploaded Files (Same as before) ---
def save_uploaded_files(uploaded_files):
    """Saves uploaded files to a temporary directory and returns their paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        try:
            # Construct the full path within the temporary directory
            file_path = os.path.join(temp_dir, uploaded_file.name)
            # Write the file contents to the temporary path
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
            logging.info(f"Saved uploaded file to temporary path: {file_path}")
        except Exception as e:
            logging.error(f"Error saving uploaded file {uploaded_file.name}: {e}", exc_info=True)
            st.error(f"Error saving file: {uploaded_file.name}")
    return file_paths, temp_dir


# --- Sidebar for File Upload, API Key, and Processing ---
with st.sidebar:
    st.header("1. Enter OpenAI API Key")
    # Get OpenAI API key from user input
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/api-keys",
        value=st.session_state.get("openai_api_key", "") # Use session state to prefill if already entered
    )
    # Update session state when input changes
    if api_key_input:
        st.session_state.openai_api_key = api_key_input
    # Display a warning if the key is still missing after potential input
    elif not st.session_state.openai_api_key:
         st.warning("Please enter your OpenAI API Key to proceed.")


    st.header("2. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, XPT, SAS7BDAT, or XLSX files", # Updated label
        accept_multiple_files=True,
        type=["pdf", "xpt", "sas7bdat", "xlsx"] # Added "xlsx" to the list of allowed types
    )

    st.header("3. Select OpenAI Model")
    # Select OpenAI model
    openai_model = st.selectbox(
        "Select OpenAI Model",
        options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], # Common OpenAI models
        index=0, # Default to gpt-4o
    )

    st.header("4. Process Files")
    process_button = st.button("Process Uploaded Files")

    # Check for API key before enabling/running processing
    if not st.session_state.openai_api_key:
        st.sidebar.error("OpenAI API Key is required to process files.")
        # Optionally disable the button if key is missing - Streamlit handles this implicitly by checking condition below
        # process_button = False # Not strictly needed due to check below

    if process_button and uploaded_files and st.session_state.openai_api_key:
        # Proceed with processing only if key and files are present
        st.session_state.files_processed = False
        st.session_state.messages = []
        st.session_state.qa_chain = None

        with st.spinner(f"Processing {len(uploaded_files)} files with {openai_model}... This may take a while."):
            try:
                file_paths, temp_dir = save_uploaded_files(uploaded_files)

                if file_paths:
                    logging.info("Loading documents...")
                    # load_documents function (from chatbot_core.py) now handles XLSX
                    documents = load_documents(file_paths)

                    if documents:
                        logging.info(f"Building QA chain with OpenAI model '{openai_model}'...")
                        # Pass the API key from session state and selected model
                        st.session_state.qa_chain = build_qa_chain(
                            documents,
                            openai_api_key=st.session_state.openai_api_key, # Use key from session state
                            openai_model_name=openai_model
                        )

                        if st.session_state.qa_chain:
                            st.session_state.files_processed = True
                            st.success(f"Processed {len(documents)} document sections. Ready to chat!")
                            logging.info("QA chain successfully built and stored in session state.")
                        else:
                             # Error message specific to potential API key issues
                             st.error("Failed to build the QA chain. Please check your API key validity and account status.")
                             logging.error("QA chain building returned None, potentially due to API key issue.")

                    else:
                        st.error("No documents could be loaded from the uploaded files. Check file content and format.")
                        logging.error("load_documents returned an empty list.")
                else:
                    st.error("Could not save uploaded files for processing.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                logging.error(f"Error during file processing or chain building: {e}", exc_info=True)

    # Handle cases where the button is clicked but conditions aren't met
    elif process_button and not st.session_state.openai_api_key:
         st.warning("Please enter your OpenAI API key above.")
    elif process_button and not uploaded_files:
        st.warning("Please upload at least one file.")


# --- Main Chat Interface ---

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display source documents if available
        if "source_docs" in message and message["source_docs"]:
             with st.expander("View Sources"):
                 for i, doc in enumerate(message["source_docs"]):
                     # Display source filename and sheet name if available (for Excel)
                     source_info = doc.metadata.get('source', 'N/A')
                     sheet_info = doc.metadata.get('sheet')
                     if sheet_info:
                         source_info += f" (Sheet: {sheet_info})"
                     st.write(f"**Source {i+1} ({source_info})**")
                     st.caption(doc.page_content[:500] + "...") # Show snippet


# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if ready to chat
    if not st.session_state.openai_api_key: # Check if key was entered
         st.warning("Please enter your OpenAI API Key in the sidebar.")
    elif not st.session_state.files_processed:
        st.warning("Please upload and process files using the sidebar.")
    elif not st.session_state.qa_chain:
         st.error("The QA system is not ready. Please ensure files were processed successfully.")
    else:
        # Generate response if ready
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            source_documents = []
            try:
                with st.spinner("Asking OpenAI..."):
                    # Prepare chat history
                    chat_history_tuples = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            human_msg = msg["content"]
                        elif msg["role"] == "assistant":
                            ai_msg = msg["content"]
                            chat_history_tuples.append((human_msg, ai_msg))

                    logging.info(f"Invoking QA chain with question: {prompt}")
                    result = st.session_state.qa_chain({
                        "question": prompt,
                        "chat_history": chat_history_tuples
                    })
                    logging.info("QA chain invocation complete.")

                    full_response = result.get("answer", "Sorry, I couldn't generate an answer.")
                    source_documents = result.get("source_documents", [])
                    message_placeholder.markdown(full_response)

                    logging.info(f"LLM Response: {full_response}")
                    if source_documents:
                         logging.info(f"Retrieved {len(source_documents)} source documents.")

            except Exception as e:
                full_response = f"An error occurred while contacting OpenAI: {e}"
                message_placeholder.error(full_response)
                logging.error(f"Error during QA chain invocation with OpenAI: {e}", exc_info=True)

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "source_docs": source_documents
            })
            # Rerun to potentially show sources
            st.rerun()
