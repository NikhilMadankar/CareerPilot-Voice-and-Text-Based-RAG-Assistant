import os
import hashlib
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

from rag.CareerGuideAssistant import CareerGuideAssistant
from career_comparison import is_comparison_query, extract_two_course_names, compare_courses
import voice_service as vs

eval_data = []
MAX_EVAL_Q = 8

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize CareerGuide AI assistant
career_guide = CareerGuideAssistant()

# Silence threshold
SILENCE_THRESHOLD = 2000


def is_silence(data, max_amplitude_threshold=SILENCE_THRESHOLD):
    return np.max(np.abs(data)) <= max_amplitude_threshold


def log_ragas_entry(question, answer, contexts):
    """Log QA pairs into ragas_eval.json for evaluation benchmarking."""
    global eval_data
    eval_data.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ""
    })

    try:
        with open("ragas_eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=4, ensure_ascii=False)
        print("✅ ragas_eval.json updated with", len(eval_data), "entries.")
    except Exception as e:
        print(f"Error logging ragas entry: {e}")


def get_career_response(query):
    """
    Route to structured comparison if query asks for a two-course comparison;
    otherwise query the standard RAG pipeline.
    Returns: {"answer": str, "retrieved_chunks": [str, ...], "sources": [dict, ...]}
    """
    if is_comparison_query(query):
        name_a, name_b = extract_two_course_names(query)
        if name_a and name_b:
            result = compare_courses(name_a, name_b)
            if result["ok"]:
                answer = career_guide.generate_direct(result["comparison_prompt"])
                sources = [
                    {
                        "course_name": result["course_a"]["course_name"],
                        "category": result["course_a"]["category"],
                        "page_number": result["course_a"]["page_number"]
                    },
                    {
                        "course_name": result["course_b"]["course_name"],
                        "category": result["course_b"]["category"],
                        "page_number": result["course_b"]["page_number"]
                    },
                ]
                return {
                    "answer": answer,
                    "retrieved_chunks": [result["course_a"]["content"], result["course_b"]["content"]],
                    "sources": sources,
                }
            else:
                print(f"⚠️ Comparison fallback: {result['error']}")

    return career_guide.interact_with_llm(query)


# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_text():
    """Text-based chat endpoint."""
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    result = get_career_response(query)
    log_ragas_entry(query, result['answer'], result['retrieved_chunks'])

    return jsonify({
        "query": query,
        "answer": result['answer'],
        "retrieved_chunks": result['retrieved_chunks'],
        "sources": result.get('sources', [])
    })


@app.route("/voice", methods=["POST"])
def chat_voice():
    """Voice-based chat endpoint (WAV upload)."""
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['file']
    temp_path = "temp_input.wav"
    file.save(temp_path)

    # Transcribe audio with Whisper
    try:
        transcription = vs.transcribe_audio(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Failed to transcribe audio: {e}"}), 500

    if not transcription:
        return jsonify({"error": "Audio contains no detectable speech"}), 400

    result = get_career_response(transcription)
    answer = result['answer']
    retrieved_chunks = result['retrieved_chunks']
    sources = result.get('sources', [])

    log_ragas_entry(transcription, answer, retrieved_chunks)

    # Generate TTS audio file in static/audio
    os.makedirs("static/audio", exist_ok=True)
    hash_str = hashlib.md5(answer.encode('utf-8')).hexdigest()[:12]
    tts_filename = f"{hash_str}.mp3"
    tts_path = os.path.join("static/audio", tts_filename)
    vs.play_text_to_file(answer, tts_path)

    return jsonify({
        "query": transcription,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "tts_audio_path": f"/static/audio/{tts_filename}"
    })


if __name__ == "__main__":
    print("🚀 Starting CareerGuide AI Server on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
