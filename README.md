# RAG-Based Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about them. Built with Groq LLaMA 3.3, ChromaDB, and Streamlit.

---

## Features

- Upload one or multiple PDF documents
- Semantic search using sentence-transformers embeddings
- Conversational memory — remembers previous questions in the session
- Source citations — shows exactly which document and page the answer came from
- Fast responses powered by Groq's LLaMA 3.3 70B model
- Clean, interactive UI built with Streamlit

---

## Architecture

```
User Question
      │
      ▼
Sentence Transformer (all-MiniLM-L6-v2)
      │
      ▼ Question Embedding
ChromaDB Vector Store ──► Top-4 Relevant Chunks
                                │
                                ▼
                    Groq LLaMA 3.3 70B + System Prompt
                                │
                                ▼
                        Answer + Source Citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq LLaMA 3.3 70B (free API) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local) |
| Vector Database | ChromaDB (local) |
| PDF Processing | PyPDF |
| Frontend | Streamlit |
| Environment | Python 3.11 |

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/PrathiAsheka/rag-chatbot.git
cd rag-chatbot
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How to Use

1. Upload one or more PDF files using the sidebar
2. Click **Process Documents** and wait for the embeddings to load
3. Type your question in the chat box
4. The chatbot will answer using only the content from your documents
5. Expand **Sources** to see which pages the answer came from

---

## Project Structure

```
rag-chatbot/
│
├── app.py              # Streamlit UI and session management
├── rag_pipeline.py     # Core RAG logic (embed, retrieve, generate)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

---

## Future Improvements

- [ ] Hybrid search (vector + keyword BM25)
- [ ] Deploy to Hugging Face Spaces for public access
- [ ] RAGAS evaluation scoring
- [ ] Support for more file types (DOCX, TXT, CSV)
- [ ] Docker deployment

---

## Author

**Prathi Asheka**
- GitHub: [@PrathiAsheka](https://github.com/PrathiAsheka)

---

## License

This project is open source and available under the [MIT License](LICENSE).
