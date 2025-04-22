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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(file_paths):
    """
    Loads documents from a list of file paths, supporting PDF, XPT, and SAS7BDAT.
    (Same implementation as the Ollama version)

    Args:
        file_paths (list): A list of paths to the files.

    Returns:
        list: A list of LangChain Document objects.
    """
    documents = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        logging.info(f"Processing file: {file_path} (Type: {file_extension})")
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    page.metadata['source'] = os.path.basename(file_path)
                    page.metadata.pop('page', None)
                documents.extend(pages)
                logging.info(f"Successfully loaded {len(pages)} pages from PDF: {os.path.basename(file_path)}")

            elif file_extension in [".xpt", ".sas7bdat"]:
                df = pd.read_sas(file_path)
                logging.info(f"Successfully read SAS file: {os.path.basename(file_path)} with shape {df.shape}")
                # Convert each row to a string Document
                for index, row in df.iterrows():
                    row_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    doc = Document(
                        page_content=f"Row {index} from {os.path.basename(file_path)}: {row_content}",
                        metadata={"source": os.path.basename(file_path), "row": index}
                    )
                    documents.append(doc)
                logging.info(f"Converted {len(df)} rows from SAS file {os.path.basename(file_path)} to Documents.")

            else:
                logging.warning(f"Unsupported file type: {file_extension} for file {file_path}")

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}", exc_info=True)

    logging.info(f"Total documents loaded: {len(documents)}")
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
        return None

    try:
        # 1. Chunking (Same as before)
        logging.info("Starting text splitting...")
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_docs)} chunks.")

        if not split_docs:
            logging.error("Text splitting resulted in zero chunks.")
            return None

        # 2. Embeddings (Using local HuggingFace model)
        # To use OpenAI embeddings instead:
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        logging.info(f"Loading embedding model: {embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        logging.info("Embedding model loaded.")

        # 3. Vector Store (FAISS - Same as before)
        logging.info("Creating FAISS vector store...")
        db = FAISS.from_documents(split_docs, embeddings)
        logging.info("FAISS vector store created successfully.")

        # 4. Retriever (Same as before)
        logging.info("Creating retriever...")
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )
        logging.info("Retriever created.")

        # 5. Large Language Model (Using OpenAI)
        logging.info(f"Initializing ChatOpenAI with model: {openai_model_name}")
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=openai_model_name,
            temperature=0.7 # Adjust temperature as needed
            )
        logging.info("ChatOpenAI initialized.")

        # 6. Conversational Retrieval Chain (Same setup, different LLM)
        logging.info("Building ConversationalRetrievalChain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
        logging.info("ConversationalRetrievalChain built successfully.")
        return qa_chain

    except Exception as e:
        # Catch potential authentication errors specifically
        if "AuthenticationError" in str(type(e)):
             logging.error(f"OpenAI Authentication Error: Please check your API key. Details: {e}", exc_info=True)
             st.error("OpenAI Authentication Error: Invalid API Key provided.") # Show error in UI
             return None # Prevent further execution
        else:
            logging.error(f"Error building QA chain: {e}", exc_info=True)
            return None
