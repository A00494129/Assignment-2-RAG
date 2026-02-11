import os
import glob

try:
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import JinaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_classic.chains import RetrievalQA
except ImportError as e:
    print(f"Error: Missing dependency ({e}).")
    print("Please ensure you are using the correct Python environment (virtualenv).")
    print("Try running: .venv\\Scripts\\python.exe rag_system.py")
    exit(1)

# --- SETUP KEYS ---
# Keys are now loaded from .env
# Ensure OPENAI_API_KEY and JINA_API_KEY are set in .env

# 1. Document Loading (Requirement: from data/ directory)
# Bonus: Support loading multiple PDF files
def load_documents(directory="data"):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        all_documents.extend(loader.load())
    return all_documents

# 2. Text Splitting (Requirement: RecursiveCharacterTextSplitter, 1000/200)
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 3. Embedding & Storage (Requirement: ChromaDB & Jina AI)
def create_vector_store(texts):
    # Using Jina Embeddings
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
         raise ValueError("JINA_API_KEY not found in environment variables")

    embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en")
    
    # Persist directory for ChromaDB
    persist_directory = "./chroma_db"
    
    # Initialize Chroma
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    return vector_store

# 4. RAG Chain (Requirement: RetrievalQA & OpenAI GPT)
def setup_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True # Bonus: Source Citations
    )
    return qa_chain

def run_test_queries(qa_chain):
    QUERIES = [
        "What is Crosswalk guards?",
        "What to do if moving through an intersection with a green signal?",
        "What to do when approached by an emergency vehicle?"
    ]
    
    os.makedirs("output", exist_ok=True)
    
    with open("output/results.txt", "w", encoding="utf-8") as f:
        for query in QUERIES:
            print(f"Processing Test Query: {query}")
            try:
                result = qa_chain.invoke({"query": query})
                answer = result["result"]
                
                # Extract citations for bonus points
                sources = []
                for doc in result.get("source_documents", []):
                    page = doc.metadata.get('page', 'unknown')
                    source = doc.metadata.get('source', 'unknown')
                    sources.append(f"{source} (Page {page})")
                
                # Deduplicate sources
                unique_sources = list(set(sources))
                sources_str = ", ".join(unique_sources)
                
                output = f"Query: {query}\nAnswer: {answer}\nSource: {sources_str}\n{'-'*20}\n"
                print(output)
            except Exception as e:
                output = f"Query: {query}\nError: {str(e)}\n{'-'*20}\n"
                print(f"Error processing query '{query}': {e}")
            
            f.write(output)
    print("Test queries complete. Results saved to output/results.txt")

def interactive_mode(qa_chain):
    print("\n--- RAG System Interactive Mode ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        query = input("Enter your question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            
            print(f"\nAnswer: {answer}")
            
            # Print citations
            print("\nSources:")
            seen_sources = set()
            for doc in result.get("source_documents", []):
                page = doc.metadata.get('page', 'unknown')
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                citation = f"{source} - Page {page}"
                if citation not in seen_sources:
                    print(f"- {citation}")
                    seen_sources.add(citation)
            print("-" * 30 + "\n")
            
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    print("Initializing RAG System...")
    
    # Check for data
    if not os.path.exists("data"):
        print("Error: 'data' directory not found.")
        exit(1)
        
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("No PDF documents found in 'data/' directory.")
        exit(1)
        
    print(f"Loaded {len(docs)} pages.")
    
    print("Splitting text...")
    texts = split_documents(docs)
    print(f"Split into {len(texts)} chunks.")
    
    print("Creating/Loading vector store...")
    vector_store = create_vector_store(texts)
    
    print("Setting up RAG chain...")
    qa_chain = setup_rag_chain(vector_store)
    
    # Run required test queries
    run_test_queries(qa_chain)
    
    # Start interactive mode
    interactive_mode(qa_chain)