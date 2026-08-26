import json
import os
import socket
import sys
import warnings

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.llms.groq import Groq
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import QueryBundle

load_dotenv(override=True)
warnings.filterwarnings("ignore")


def is_port_open(host="localhost", port=6333, timeout=0.5):
    """Check if Qdrant server is reachable without hanging."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class CareerGuideAssistant:
    """
    RAG Assistant for NCERT Career & Course Guidance.
    Orchestrates Qdrant Vector Store + HuggingFace Embeddings + Groq LLM (Llama 3.1) via LlamaIndex.
    """

    def __init__(self, qdrant_url=None, collection_name="career_guide_db"):
        self._collection_name = collection_name
        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")

        # Fast connection check
        if "localhost" in self._qdrant_url and not is_port_open("localhost", 6333):
            storage_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qdrant_storage"))
            print(f"[CareerGuideAssistant] Qdrant server not running at 6333. Using local disk storage at {storage_path}")
            self._client = QdrantClient(path=storage_path)
        else:
            try:
                self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False, timeout=1.0)
                self._client.get_collections()
                print(f"[CareerGuideAssistant] Connected to Qdrant server at {self._qdrant_url}")
            except Exception:
                storage_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qdrant_storage"))
                print(f"[CareerGuideAssistant] Falling back to local disk storage at {storage_path}")
                self._client = QdrantClient(path=storage_path)

        # Groq LLM
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key or groq_key == "Your Api Key":
            print("[CareerGuideAssistant] Warning: GROQ_API_KEY is not configured in .env. Please set a valid key for LLM responses.")

        self._llm = Groq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=groq_key if groq_key and groq_key != "Your Api Key" else "dummy_key",
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1024"))
        )

        # Embedding model
        self._embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )

        # Apply to global settings
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        # Index and chat engine
        self._index = None
        self._create_kb()
        if self._index is not None:
            self._create_chat_engine()
        else:
            print("[CareerGuideAssistant] Failed to create knowledge base. Chat engine not initialized.")

    def _kb_json_path(self):
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, "career_courses.json")

    def _create_kb(self):
        """Load structured courses JSON and index into Qdrant with rich metadata."""
        try:
            json_path = self._kb_json_path()
            if not os.path.exists(json_path):
                print(f"[CareerGuideAssistant] Knowledge base JSON file not found at {json_path}")
                return

            with open(json_path, "r", encoding="utf-8") as f:
                courses = json.load(f)

            vector_store = QdrantVectorStore(client=self._client, collection_name=self._collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Check if collection already exists in Qdrant to avoid re-embedding every time
            existing_collections = [c.name for c in self._client.get_collections().collections]
            if self._collection_name in existing_collections:
                self._index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
                print(f"[CareerGuideAssistant] Loaded existing vector collection '{self._collection_name}' from Qdrant.")
                return

            # Build index and upload embeddings into Qdrant
            documents = []
            for c in courses:
                content = c.get("content", "")
                metadata = {
                    "course_id": c.get("id", ""),
                    "course_name": c.get("course_name", "Unknown"),
                    "category": c.get("category", "General"),
                    "toc_number": c.get("toc_number", 0),
                    "page_number": c.get("page_number", 0),
                }
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)

            self._index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            print(f"[CareerGuideAssistant] Created knowledge base ({len(documents)} courses) in Qdrant collection '{self._collection_name}'.")
        except Exception as e:
            print(f"[CareerGuideAssistant] Error while creating knowledge base: {e}")

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def retrieve_only(self, query, top_k=5):
        """
        Retrieval WITHOUT triggering an LLM chat call.
        Returns: {"retrieved_chunks": [<str>, ...], "sources": [<dict>, ...]}
        """
        if not query or not query.strip() or self._index is None:
            return {"retrieved_chunks": [], "sources": []}

        try:
            query_embedding = self._embed_model.get_text_embedding(query)
            if query_embedding is None:
                return {"retrieved_chunks": [], "sources": []}

            retriever = self._index.as_retriever(similarity_top_k=top_k)
            bundle = QueryBundle(query_str=query, embedding=query_embedding)
            nodes = retriever.retrieve(bundle)

            retrieved_texts = []
            sources = []
            seen_courses = set()

            for n in nodes:
                text = None
                try:
                    text = n.node.get_text()
                except Exception:
                    text = getattr(n, "text", None) or getattr(n, "content", None) or getattr(n, "page_content", None)
                if not text:
                    text = str(n)
                retrieved_texts.append(text)

                meta = getattr(n.node, "metadata", {}) or {}
                c_name = meta.get("course_name", "Career Guide")
                if c_name not in seen_courses:
                    seen_courses.add(c_name)
                    sources.append({
                        "course_name": c_name,
                        "category": meta.get("category", "General"),
                        "page_number": meta.get("page_number", "N/A"),
                    })

            return {"retrieved_chunks": retrieved_texts, "sources": sources}
        except Exception as e:
            print(f"[CareerGuideAssistant] Error during retrieval: {e}")
            return {"retrieved_chunks": [], "sources": []}

    def interact_with_llm(self, customer_query, top_k=5):
        """
        Standard RAG interaction.
        Returns: {"answer": str, "retrieved_chunks": [str, ...], "sources": [dict, ...]}
        """
        try:
            if not customer_query or not customer_query.strip():
                return {"answer": "", "retrieved_chunks": [], "sources": []}

            retrieval = self.retrieve_only(customer_query, top_k=top_k)

            # Generate response via Chat Engine
            if self._chat_engine:
                response = self._chat_engine.chat(customer_query)
                answer_text = getattr(response, "response", None) or str(response)
            else:
                answer_text = "Chat engine is not initialized."

            return {
                "answer": answer_text,
                "retrieved_chunks": retrieval["retrieved_chunks"],
                "sources": retrieval["sources"],
            }
        except Exception as e:
            return {"answer": f"Error in interaction: {e}", "retrieved_chunks": [], "sources": []}

    def generate_direct(self, prompt: str) -> str:
        """Send a formatted prompt straight to the LLM, bypassing retrieval."""
        try:
            response = self._llm.complete(prompt)
            return getattr(response, "text", None) or str(response)
        except Exception as e:
            return f"Error generating response: {e}"

    @property
    def _prompt(self):
        return """
        You are CareerGuide AI, a friendly, encouraging, and knowledgeable AI educational counselor
        helping students (primarily after 10+2 / 12th grade) and aspirants explore courses, college streams,
        and career pathways in India.

        You answer using the information from the NCERT/CBSE "Compendium of Academic Courses After +2" knowledge base,
        which covers eligibility criteria, course options, and institutes for 100+ streams including
        Engineering, Medical & Health Sciences, Science, Design, Arts, Law, Management, and more.

        Guidelines:
        - Be encouraging, clear, and structured like a professional school counselor.
        - When asked about a specific course, summarize: what the course involves, 10+2 eligibility requirements, degree levels (UG, PG, PhD), and notable institutes/universities.
        - When the student describes their interests (e.g. "I like biology and mathematics", "I want to work in aviation"), suggest 2-3 relevant course options with brief reasoning.
        - If the knowledge base does not contain the answer, state that honestly rather than hallucinating details.
        - Keep answers structured with bullet points and bold highlights, suitable for both reading and spoken text-to-speech.
        """
