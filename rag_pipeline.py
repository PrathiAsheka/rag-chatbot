from dotenv import load_dotenv
load_dotenv()

import os
import pypdf
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq


class RAGPipeline:
    """
    RAG Pipeline using raw libraries — no LangChain.

    Embeddings : sentence-transformers (free, local)
    LLM        : Groq LLaMA 3 (free API)
    Vector DB  : ChromaDB (free, local)
    """

    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")

        self.groq = Groq(api_key=groq_api_key)
        self.chroma = chromadb.Client()
        self.collection = None
        self.chat_history = []
        print("Pipeline ready!")

    def load_documents(self, file_paths: list) -> None:
        """Load PDFs, chunk, embed, and store in ChromaDB."""

        # Reset collection each time new docs are loaded
        try:
            self.chroma.delete_collection("rag_docs")
        except Exception:
            pass
        self.collection = self.chroma.create_collection("rag_docs")
        self.chat_history = []

        all_chunks = []
        all_metadata = []

        for path in file_paths:
            filename = os.path.basename(path)
            print(f"Loading: {filename}")

            # Extract text from PDF
            reader = pypdf.PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                # Chunk each page into smaller pieces
                chunks = self._chunk_text(text, chunk_size=500, overlap=50)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": filename,
                        "page": str(page_num + 1)
                    })

        if not all_chunks:
            raise ValueError("No text found in the uploaded PDFs.")

        print(f"Embedding {len(all_chunks)} chunks...")

        # Embed all chunks
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True)

        # Store in ChromaDB
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadata,
            ids=[f"chunk_{i}" for i in range(len(all_chunks))]
        )

        print(f"Stored {len(all_chunks)} chunks in vector DB.")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def query(self, question: str, chat_history: list = None) -> dict:
        """Retrieve relevant chunks and generate an answer with Groq."""

        if not self.collection:
            raise ValueError("No documents loaded. Please upload PDFs first.")

        # Embed the question
        question_embedding = self.embedder.encode([question]).tolist()

        # Retrieve top-4 relevant chunks
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=4
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        context = "\n\n".join(chunks)

        # Build source list
        sources = []
        seen = set()
        for chunk, meta in zip(chunks, metadatas):
            key = (meta["source"], meta["page"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": meta["source"],
                    "page": meta["page"],
                    "content": chunk
                })

        # Build messages for Groq
        system_prompt = """You are a helpful assistant. Answer the user's question
using ONLY the context provided below. Be clear and concise.
If the answer is not in the context, say:
"I don't have enough information in the uploaded documents to answer that."
Do NOT use outside knowledge.

Context:
""" + context

        messages = [{"role": "system", "content": system_prompt}]

        # Add last 4 exchanges for memory
        for msg in self.chat_history[-8:]:
            messages.append(msg)

        messages.append({"role": "user", "content": question})

        # Call Groq
        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        # Save to history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "sources": sources}
