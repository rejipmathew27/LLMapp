import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Keeping local embeddings for now
# from langchain_openai import OpenAIEmbeddings # Option to switch to OpenAI embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI # Import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
import logging
import io # Needed for capturing DataFrame string representation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(file_paths):
    """
    Loads documents from a list of file paths, supporting PDF, XPT, and SAS7BDAT.
    Also generates and includes summary statistics for SAS files.

    Args:
        file_paths (list): A list of paths to the files.

    Returns:
        list: A list of LangChain Document objects, including summaries for SAS files.
    """
    documents = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        base_filename = os.path.basename(file_path)
        logging.info(f"Processing file: {base_filename} (Type: {file_extension})")
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    page.metadata['source'] = base_filename
                    page.metadata.pop('page', None) # Remove default page number if not needed
                documents.extend(pages)
                logging.info(f"Successfully loaded {len(pages)} pages from PDF: {base_filename}")

            elif file_extension in [".xpt", ".sas7bdat"]:
                # Load SAS file into a pandas DataFrame
                df = pd.read_sas(file_path)
                logging.info(f"Successfully read SAS file: {base_filename} with shape {df.shape}")

                # --- Generate and Add Summary Statistics Document ---
                try:
                    logging.info(f"Generating summary statistics for {base_filename}...")
                    # Use describe() to get statistics for all columns (numeric and categorical)
                    summary_stats = df.describe(include='all').to_string() # Convert summary to string

                    # Create a separate Document for the summary statistics
                    summary_doc = Document(
                        page_content=f"Summary Statistics for {base_filename}:\n\n{summary_stats}",
                        metadata={"source": base_filename, "content_type": "summary_statistics"}
                    )
                    documents.append(summary_doc)
                    logging.info(f"Successfully added summary statistics document for {base_filename}.")
                except Exception as summary_e:
                     logging.error(f"Could not generate summary statistics for {base_filename}: {summary_e}", exc_info=True)
                # --- End of Summary Statistics ---

                # --- Add Document per Row (Optional - provides detailed data) ---
                # Convert each row to a string Document
                for index, row in df.iterrows():
                    # Handle potential mixed types or errors during string conversion within a row
                    try:
                        row_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    except Exception as row_e:
                        row_content = f"Error converting row {index} to string: {row_e}"
                        logging.warning(f"Error converting row {index} in {base_filename} to string: {row_e}")

                    doc = Document(
                        page_content=f"Row {index} from {base_filename}: {row_content}",
                        metadata={"source": base_filename, "row": index, "content_type": "row_data"}
                    )
                    documents.append(doc)
                logging.info(f"Converted {len(df)} rows from SAS file {base_filename} to Documents.")
                # --- End of Row Data ---

            else:
                logging.warning(f"Unsupported file type: {file_extension} for file {file_path}")

        except pd.errors.ParserError as pe:
             logging.error(f"Pandas parsing error processing file {base_filename}: {pe}. Is the file format correct?", exc_info=True)
        except Exception as e:
            logging.error(f"Error processing file {base_filename}: {e}", exc_info=True)

    logging.info(f"Total documents loaded (including summaries): {len(documents)}")
    return documents

def build_qa_chain(documents, openai_api_key, openai_model_name="gpt-3.5-turbo", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds the RAG (Retrieval-Augmented Generation) chain using OpenAI.

    Args:
        documents (list): List of LangChain Document objects.
        openai_api_key (str): The OpenAI API key.
        openai_model_name (str): The name of the OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").
        embedding_model_name (str): The name of the sentence-transformer model for embeddings.
                                     (Set to None if using OpenAIEmbeddings).

    Returns:
        ConversationalRetrievalChain: The ready-to-use question-answering chain.
        None: If no documents were provided or an error occurred.
    """
    if not documents:
        logging.error("No documents provided to build the QA chain.")
        return None
    if not openai_api_key:
        logging.error("OpenAI API Key is required.")
        # Note: The Streamlit app should ideally prevent calling this without a key
        # but this check remains as a safeguard in the core logic.
        return None

    try:
        # 1. Chunking (Same as before)
        logging.info("Starting text splitting...")
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n\n", # Try splitting on paragraphs first
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_docs)} chunks.")

        if not split_docs:
            logging.error("Text splitting resulted in zero chunks. Check document content and splitter settings.")
            return None

        # 2. Embeddings (Using local HuggingFace model)
        # To use OpenAI embeddings instead:
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        logging.info(f"Loading embedding model: {embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        logging.info("Embedding model loaded.")

        # 3. Vector Store (FAISS - Same as before)
        logging.info("Creating FAISS vector store...")
        # This step can take significant time and memory depending on the number of chunks
        db = FAISS.from_documents(split_docs, embeddings)
        logging.info("FAISS vector store created successfully.")

        # 4. Retriever (Same as before)
        logging.info("Creating retriever...")
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5} # Retrieve top 5 relevant chunks
        )
        logging.info("Retriever created.")

        # 5. Large Language Model (Using OpenAI)
        logging.info(f"Initializing ChatOpenAI with model: {openai_model_name}")
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=openai_model_name,
            temperature=0.7 # Adjust temperature as needed (0=deterministic, >1=more creative)
            )
        logging.info("ChatOpenAI initialized.")

        # 6. Conversational Retrieval Chain (Same setup, different LLM)
        logging.info("Building ConversationalRetrievalChain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            # Consider adding memory if you want conversation context beyond the immediate question/answer pair
            # from langchain.memory import ConversationBufferMemory
            # memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
        logging.info("ConversationalRetrievalChain built successfully.")
        return qa_chain

    # Catch specific exceptions if needed, e.g., related to FAISS or Embeddings
    except ImportError as ie:
         logging.error(f"Import error during QA chain build: {ie}. Ensure all dependencies are installed.", exc_info=True)
         # You might want to raise this or return None depending on desired handling
         return None
    except Exception as e:
        # Catch potential authentication errors specifically
        # Check if the exception or its cause is related to authentication
        if "AuthenticationError" in str(type(e)) or (hasattr(e, 'cause') and "AuthenticationError" in str(type(e.cause))):
             logging.error(f"OpenAI Authentication Error: Please check your API key. Details: {e}", exc_info=False) # Avoid logging key details if possible
             # The Streamlit app should display an error based on the None return value
             return None # Prevent further execution
        else:
            logging.error(f"Error building QA chain: {e}", exc_info=True)
            return None
