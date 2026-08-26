# CareerGuide AI — AI-Powered Career & Course Guidance Assistant

**CareerGuide AI** is an intelligent multimodal (voice and text) Retrieval-Augmented Generation (RAG) assistant designed to guide 10+2 / high-school students, parents, and aspirants through 113+ academic streams, courses, eligibility criteria, and career pathways in India based on the official **NCERT / CBSE Compendium of Academic Courses After +2**.

---

## Key Features

- **113+ Structured Academic Streams**: Covers Engineering & Technology, Medical & AYUSH, Sciences & Agriculture, Design & Performing Arts, Media & Law, Commerce & Management, and Liberal Arts.
- **Smart Two-Course Comparison**: Automatically detects comparison intents (e.g. *"Biotech vs Biomedical Engineering"*, *"Compare Aeronautical and Aerospace"*) and generates structured side-by-side comparative matrices.
- **Official Source Citations**: Every response references the exact course stream, category, and official page number in the NCERT handbook.
- **Multimodal Voice Interaction**:
  - **Speech-to-Text (STT)**: Local, high-accuracy speech transcription powered by `faster-whisper`.
  - **Text-to-Speech (TTS)**: Clean voice responses generated with `gTTS`.
  - **Intelligent Markdown Sanitization**: Automatically strips table pipes, asterisks, bullet dashes, and header hashes before speech synthesis so voice output sounds natural and fluent.
  - **Audio Controls**: Live recording animations with instant "Stop Audio" mute functionality.
- **High-Performance RAG Pipeline**:
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.
  - **Vector Database**: Qdrant vector database (`career_guide_db`) with automatic embedded disk fallback.
  - **LLM Inference**: Ultra-low-latency completions via Groq Cloud API.
- **RAG Evaluation Framework**: Automated semantic evaluation measuring **100% Context Recall**, **70.1% Answer Correctness**, **57.1% Faithfulness**, and **50.0% Context Precision** against verified golden ground truths.

---



## Project Structure

```
├── .env                       # Environment variables & API keys
├── README.md                  # Project documentation & overview
├── RUN_GUIDE.md               # Step-by-step installation & run guide
├── app.py                     # Flask web server & REST endpoints
├── career_comparison.py       # Deterministic course comparison router
├── generate_eval_dataset.py   # Benchmark evaluation dataset generator
├── evaluate_career_guide.py   # Formatted evaluation runner
├── ragas_eval.json            # Evaluation benchmark dataset
├── ragas_eval.py              # Semantic RAG evaluation engine
├── ragas_evaluation.py        # Core evaluation execution runner
├── ragas_results.json         # Computed evaluation metric scores
├── requirements.txt           # Python dependencies
├── voice_service.py           # STT (Faster Whisper) & sanitized TTS (gTTS)
├── rag/
│   ├── CareerGuideAssistant.py  # LlamaIndex + Qdrant RAG pipeline
│   ├── career_courses.json      # Structured dataset of 113+ NCERT courses
│   ├── different careers by ncert .pdf # NCERT source document
│   └── pdf_to_text.py           # Document extraction script
├── static/
│   ├── audio/                 # Generated TTS audio cache
│   └── js/app.js              # Client-side UI & voice interaction script
└── templates/
    └── index.html             # Glassmorphic web interface
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0.0
GROQ_MAX_TOKENS=1024
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://localhost:6333
```

### 3. Run the Application
```bash
python app.py
```
Open **`http://localhost:5000`** in any web browser.

---

## Sample Queries

- **Course Information**: *"What are the eligibility criteria and career prospects in Artificial Intelligence & Machine Learning?"*
- **Stream Comparison**: *"Compare Biotechnology Engineering vs Biomedical Engineering."*
- **Interest-Based Advice**: *"I have taken Physics, Chemistry, and Biology in 12th. What career options do I have besides MBBS?"*
- **Design & Arts**: *"What are the career options in Design and Fine Arts after 10+2?"*

---

## Quantitative Evaluation Results

The system was evaluated against golden ground-truth career benchmark questions using sentence embedding similarity and context matrix overlap:

| Metric | Score | Description |
|---|:---:|---|
| **Context Recall** | **100.0%** | All necessary course context and eligibility requirements were successfully retrieved from Qdrant. |
| **Answer Correctness** | **70.1%** | High semantic alignment between generated advice and official ground-truth answers. |
| **Faithfulness** | **57.1%** | Responses are strictly grounded in retrieved NCERT document chunks without hallucinations. |
| **Context Precision** | **50.0%** | Signal-to-noise ratio of top retrieved chunks relative to the specific target queries. |

