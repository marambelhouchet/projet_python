import os
import shutil
import requests # Added to check Ollama status

# --- Langchain specific imports ---
# Document Loaders
from langchain_community.document_loaders import UnstructuredMarkdownLoader
# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings Model (Ollama)
from langchain_community.embeddings import OllamaEmbeddings
# Vector Store (FAISS)
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# Path to the single Markdown file containing all disease info
# Assumes this script is run from the 'backend_django' folder
DOCUMENTS_PATH = "agri_documents/disease_database.md"
VECTORSTORE_PATH = "vectorstore" # Folder where the FAISS index will be saved
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Ollama embedding model to use (e.g., nomic-embed-text, mxbai-embed-large)
# --- End Configuration ---

def check_ollama_status():
    """Checks if the Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        response.raise_for_status() # Raise an exception for bad status codes
        print("✓ Ollama server is running.")
        return True
    except requests.exceptions.RequestException as e:
        print("\n❌ ERROR: Ollama server not reachable at http://localhost:11434.")
        print(f"   Details: {e}")
        print("   Please start Ollama server (e.g., 'ollama serve') before running this script.")
        return False
    except Exception as e:
        print(f"\n❌ ERROR checking Ollama status: {e}")
        return False

def check_embedding_model_availability(model_name):
    """Checks if the specified embedding model is available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        available = any(model['name'].startswith(model_name + ':') for model in models)
        if available:
            print(f"✓ Embedding model '{model_name}' found in Ollama.")
            return True
        else:
            print(f"\n⚠️ WARNING: Embedding model '{model_name}' not found in Ollama.")
            print(f"   Please pull it using: `ollama pull {model_name}`")
            return False # Continue but warn
    except requests.exceptions.RequestException as e:
        print(f"\n⚠️ WARNING: Could not verify embedding model availability. Error connecting to Ollama tags API: {e}")
        return True # Assume it exists, proceed with caution
    except Exception as e:
        print(f"\n⚠️ WARNING: Error checking model availability: {e}")
        return True # Assume it exists

def main():
    print("--- Starting Vector Database Creation ---")

    # 1. Check Ollama Status
    if not check_ollama_status():
        exit(1) # Exit if Ollama is not running

    # 2. Check Embedding Model Availability (Optional but recommended)
    check_embedding_model_availability(EMBEDDING_MODEL_NAME)

    # 3. Check Document Path
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"\n❌ ERROR: The document file was not found at: {DOCUMENTS_PATH}")
        print("   Please create the 'agri_documents' folder and place 'disease_database.md' inside it.")
        doc_dir = os.path.dirname(DOCUMENTS_PATH)
        if not os.path.isdir(doc_dir):
            try:
                os.makedirs(doc_dir)
                print(f"   Created directory: {doc_dir}")
                print("   Please add your 'disease_database.md' file inside it.")
            except Exception as e:
                print(f"   Could not create directory {doc_dir}: {e}")
        exit(1) # Exit if document is missing

    print(f"\nLoading document from: {DOCUMENTS_PATH}")

    # 4. Load the single Markdown document
    try:
        # Using UnstructuredMarkdownLoader handles Markdown structure better
        loader = UnstructuredMarkdownLoader(DOCUMENTS_PATH, mode="elements")
        documents = loader.load()
        if not documents:
            print("❌ ERROR: No documents were loaded. Is the file empty or in an unexpected format?")
            exit(1)
        print(f"✓ Loaded {len(documents)} document section(s).")
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        exit(1)

    # 5. Split documents into chunks
    print("\nSplitting documents into chunks...")
    # RecursiveCharacterTextSplitter is good for general text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max size of each chunk
        chunk_overlap=150, # Overlap between chunks to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)
    if not split_docs:
        print("❌ ERROR: Splitting resulted in zero chunks. Check document content and chunk size.")
        exit(1)
    print(f"✓ Split into {len(split_docs)} chunks.")


    # 6. Create embeddings (using Ollama)
    print(f"\nCreating embeddings using Ollama model: '{EMBEDDING_MODEL_NAME}' (this may take a while)...")
    try:
        # Initialize the Ollama embedding model
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        # Test embedding a small text to ensure connection and model work
        print("   Testing embedding model...")
        _ = embeddings.embed_query("Test embedding.")
        print("✓ Embedding model initialized successfully.")
    except Exception as e:
        print(f"\n❌ Error initializing Ollama embeddings: {e}")
        print(f"   Make sure Ollama is running and the model '{EMBEDDING_MODEL_NAME}' is downloaded (`ollama pull {EMBEDDING_MODEL_NAME}`).")
        exit(1)

    # 7. Create FAISS vector store
    print("\nCreating FAISS vector store...")
    try:
        # Delete old vectorstore if it exists for a clean build
        if os.path.exists(VECTORSTORE_PATH):
            print(f"   Removing existing vector store at: {VECTORSTORE_PATH}")
            shutil.rmtree(VECTORSTORE_PATH)

        # Create the vector store from document chunks and embeddings
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        # Save the vector store locally
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"✓ Vector store created and saved locally at: {VECTORSTORE_PATH}")
    except Exception as e:
        print(f"❌ Error creating or saving FAISS vector store: {e}")
        exit(1)

    print("\n--- Vector Database Creation Complete ---")

if __name__ == "__main__":
    main()

