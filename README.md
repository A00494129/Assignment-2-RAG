# Document Q&A with RAG System

This project implements a Retrieval Augmented Generation (RAG) system using **LangChain** and **ChromaDB**. It loads PDF documents, creates embeddings using **Jina AI**, and allows users to ask questions about the content, which are answered by **Google Gemini (gemini-flash-latest)**.

## Assignment Details
- **Course**: Web, Mobile, Cloud app
- **Assignment**: 2 - Document Q&A with RAG
- **Objective**: Build a RAG system to answer questions from `DH-Chapter2.pdf`.

## Features
- **Document Loading**: Automatically loads all PDF files from the `data/` directory.
- **Text Splitting**: Uses `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).
- **Vector Store**: Stores embeddings locally using **ChromaDB**.
- **Interactive CLI**: Continuous question-answering loop for users.
- **Source Citations**: Provides page numbers and source filenames for every answer (Bonus).
- **Automated Testing**: Runs 3 required test queries on startup and saves results to `output/results.txt`.

## Prerequisites
- Python 3.10+
- Google API Key (Free Tier)
- Jina AI API Key

## Setup & Installation

1. **Clone the repository** (if applicable) or download the files.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   Create a `.env` file in the root directory (already included in `.gitignore`):
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   JINA_API_KEY=your_jina_api_key_here
   # OpenAI Key is no longer required
   ```

## Usage

Run the main script:
```bash
python rag_system.py
```

### What happens when you run it:
1. **Loading**: Use `PyPDFLoader` to load PDFs from `data/`.
2. **Indexing**: Splits text and creates/loads the vector index.
3. **Testing**: Automatically runs 3 predefined queries:
   - "What is Crosswalk guards?"
   - "What to do if moving through an intersection with a green signal?"
   - "What to do when approached by an emergency vehicle?"
   Results are saved to `output/results.txt`.
4. **Interactive Mode**: You can type your own questions. Type `exit` to quit.

## Troubleshooting
- **API Errors**: Ensure your `GOOGLE_API_KEY` is valid and has access to the `gemini-2.5-flash` model.
- **Quota**: Google's free tier has generous limits but can be exceeded.
