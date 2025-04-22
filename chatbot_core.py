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
    Loads documents from a list of file paths, supporting PDF, XPT, SAS7BDAT, and XLSX.
    Also generates and includes summary statistics for tabular files (SAS, Excel).

    Args:
        file_paths (list): A list of paths to the files.

    Returns:
        list: A list of LangChain Document objects, including summaries for tabular files.
    """
    documents = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        base_filename = os.path.basename(file_path)
        logging.info(f"Processing file: {base_filename} (Type: {file_extension})")
        try:
            # --- PDF Handling ---
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    page.metadata['source'] = base_filename
                    page.metadata.pop('page', None) # Remove default page number if not needed
                documents.extend(pages)
                logging.info(f"Successfully loaded {len(pages)} pages from PDF: {base_filename}")

            # --- SAS Handling (XPT, SAS7BDAT) ---
            elif file_extension in [".xpt", ".sas7bdat"]:
                df = pd.read_sas(file_path)
                logging.info(f"Successfully read SAS file: {base_filename} with shape {df.shape}")

                # Generate and Add Summary Statistics Document
                try:
                    logging.info(f"Generating summary statistics for {base_filename}...")
                    summary_stats = df.describe(include='all').to_string()
                    summary_doc = Document(
                        page_content=f"Summary Statistics for {base_filename}:\n\n{summary_stats}",
                        metadata={"source": base_filename, "content_type": "summary_statistics"}
                    )
                    documents.append(summary_doc)
                    logging.info(f"Successfully added summary statistics document for {base_filename}.")
                except Exception as summary_e:
                     logging.error(f"Could not generate summary statistics for {base_filename}: {summary_e}", exc_info=True)

                # Add Document per Row
                for index, row in df.iterrows():
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

            # --- Excel Handling (XLSX) ---
            elif file_extension == ".xlsx":
                # Read all sheets into a dictionary of DataFrames
                excel_file = pd.ExcelFile(file_path)
                all_sheets_data = excel_file.parse(sheet_name=None) # Reads all sheets
                logging.info(f"Read {len(all_sheets_data)} sheets from Excel file: {base_filename}")

                for sheet_name, df in all_sheets_data.items():
                    logging.info(f"Processing sheet '{sheet_name}' from {base_filename} with shape {df.shape}")

                    # Generate and Add Summary Statistics Document for the sheet
                    try:
                        logging.info(f"Generating summary statistics for sheet '{sheet_name}' in {base_filename}...")
                        summary_stats = df.describe(include='all').to_string()
                        summary_doc = Document(
                            page_content=f"Summary Statistics for sheet '{sheet_name}' in {base_filename}:\n\n{summary_stats}",
                            metadata={"source": base_filename, "sheet": sheet_name, "content_type": "summary_statistics"}
                        )
                        documents.append(summary_doc)
                        logging.info(f"Successfully added summary statistics document for sheet '{sheet_name}'.")
                    except Exception as summary_e:
                        logging.error(f"Could not generate summary statistics for sheet '{sheet_name}' in {base_filename}: {summary_e}", exc_info=True)

                    # Add Document per Row for the sheet
                    for index, row in df.iterrows():
                        try:
                            row_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                        except Exception as row_e:
                            row_content = f"Error converting row {index} to string: {row_e}"
                            logging.warning(f"Error converting row {index} from sheet '{sheet_name}' in {base_filename} to string: {row_e}")
                        doc = Document(
                            page_content=f"Row {index} from sheet '{sheet_name}' in {base_filename}: {row_content}",
                            metadata={"source": base_filename, "sheet": sheet_name, "row": index, "content_type": "row_data"}
                        )
                        documents.append(doc)
                    logging.info(f"Converted {len(df)} rows from sheet '{sheet_name}' in {base_filename} to Documents.")

            # --- Unsupported File Type ---
            else:
                logging.warning(f"Unsupported file type: {file_extension} for file {file_path}")

        except pd.errors.ParserError as pe:
             logging.error(f"Pandas parsing error processing file {base_filename}: {pe}. Is the file format correct?", exc_info=True)
        except ImportError as ie:
            # Specifically check if openpyxl is missing for xlsx files
            if 'openpyxl' in str(ie) and file_extension == ".xlsx":
                 logging.error(f"Missing 'openpyxl' library needed for Excel file {base_filename}. Please install it (`pip install openpyxl`).", exc_info=False)
                 # Optionally raise or inform the user via Streamlit if possible
            else:
                 logging.error(f"Import error processing file {base_filename}: {ie}", exc_info=True)
        except Exception as e:
            logging.error(f"Error processing file {base_filename}: {e}", exc_info=True)

    logging.info(f"Total documents loaded (including summaries): {len(documents)}")
    return documents

# --- build_qa_chain function remains the same as the previous version ---
# (It doesn't need modification as it operates on the list of Document objects)
def build_qa_chain(documents, openai_api_key, openai_model_name="gpt-3.5-turbo", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds the RAG (Retrieval-Augmented Generation) chain using OpenAI.
    (No changes needed here from the previous version)
    """
    if not documents:
        logging.error("No documents provided to build the QA chain.")
        return None
    if not openai_api_key:
        logging.error("OpenAI API Key is required.")
        return None

    try:
        # 1. Chunking
        logging.info("Starting text splitting...")
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n\n",
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_docs)} chunks.")

        if not split_docs:
            logging.error("Text splitting resulted in zero chunks.")
            return None

        # 2. Embeddings
        logging.info(f"Loading embedding model: {embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        logging.info("Embedding model loaded.")

        # 3. Vector Store (FAISS)
        logging.info("Creating FAISS vector store...")
        db = FAISS.from_documents(split_docs, embeddings)
        logging.info("FAISS vector store created successfully.")

        # 4. Retriever
        logging.info("Creating retriever...")
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )
        logging.info("Retriever created.")

        # 5. Large Language Model (OpenAI)
        logging.info(f"Initializing ChatOpenAI with model: {openai_model_name}")
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=openai_model_name,
            temperature=0.7
            )
        logging.info("ChatOpenAI initialized.")

        # 6. Conversational Retrieval Chain
        logging.info("Building ConversationalRetrievalChain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
        logging.info("ConversationalRetrievalChain built successfully.")
        return qa_chain

    except ImportError as ie:
         logging.error(f"Import error during QA chain build: {ie}. Ensure all dependencies are installed.", exc_info=True)
         return None
    except Exception as e:
        if "AuthenticationError" in str(type(e)) or (hasattr(e, 'cause') and "AuthenticationError" in str(type(e.cause))):
             logging.error(f"OpenAI Authentication Error: Please check your API key. Details: {e}", exc_info=False)
             return None
        else:
            logging.error(f"Error building QA chain: {e}", exc_info=True)
            return None
