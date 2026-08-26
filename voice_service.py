import os
import re
from gtts import gTTS
from faster_whisper import WhisperModel

# Config via environment variables (defaults)
_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")  # base is fast and accurate for English
_WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
_WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

_whisper_model = None


def get_whisper_model():
    """Lazy-load the Whisper model on the first transcription call."""
    global _whisper_model
    if _whisper_model is None:
        print(f"[voice_service] Loading faster-whisper model: {_MODEL_SIZE}.en on {_WHISPER_DEVICE} ...")
        _whisper_model = WhisperModel(f"{_MODEL_SIZE}.en", device=_WHISPER_DEVICE, compute_type=_WHISPER_COMPUTE_TYPE)
        print("[voice_service] faster-whisper loaded.")
    return _whisper_model


def transcribe_audio(file_path, beam_size=5):
    """
    Transcribe an audio file using faster-whisper. Accepts wav/webm/ogg/mp3.
    Returns transcription (string) or '' on failure.
    """
    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(file_path, beam_size=beam_size)
        transcription = " ".join([seg.text for seg in segments]).strip()
        return transcription
    except Exception as e:
        print("[voice_service] Transcription error:", e)
        return ""


def sanitize_text_for_tts(text: str) -> str:
    """
    Strip markdown/formatting characters that LLM answers commonly include
    (bullet dashes, headers, bold/italic asterisks, table pipes, code
    backticks, etc.) so gTTS doesn't read them out literally as
    "dash", "hash", "asterisk", "pipe", and so on.

    The DISPLAYED answer in the chat UI is left untouched -- this cleaned
    version is used ONLY as input to gTTS.
    """
    if not text:
        return text

    cleaned = text

    # Remove markdown table rows / pipe separators (| col | col |)
    cleaned = re.sub(r"\|", " ", cleaned)

    # Remove markdown headers (#, ##, ### at line start)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # Remove bold/italic markers (**text**, *text*, __text__, _text_)
    # but keep the inner text
    cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned)

    # Remove inline code / code blocks (`text`, ```text```)
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)

    # Convert bullet list markers ("- ", "* ", "+ " at line start) into
    # nothing (so gTTS doesn't say "dash"/"asterisk" for every list item)
    cleaned = re.sub(r"^[\-\*\+]\s+", "", cleaned, flags=re.MULTILINE)

    # Convert numbered list markers "1. " / "1) " to nothing extra --
    # gTTS reading "one" is fine, just drop stray punctuation duplication
    cleaned = re.sub(r"^\d+[\.\)]\s+", "", cleaned, flags=re.MULTILINE)

    # Remove standalone horizontal rule lines (---, ***, ___)
    cleaned = re.sub(r"^[\-\*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)

    # Remove leftover dash/equals runs (e.g. from markdown table separator
    # rows like |---|---| once the pipes are already stripped)
    cleaned = re.sub(r"[\-=]{2,}", " ", cleaned)

    # Remove markdown links [text](url) -> keep just the text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)

    # Collapse any remaining stray symbols commonly read aloud awkwardly
    cleaned = re.sub(r"[#\*_~|>]", " ", cleaned)

    # Collapse multiple newlines/spaces left behind by the above
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = re.sub(r"\n", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    return cleaned.strip()


def play_text_to_file(text, out_file_path, language='en', slow=False):
    """
    Generate an MP3 using gTTS and save directly to out_file_path.
    Text is sanitized first so markdown symbols aren't read aloud.
    """
    try:
        safe_text = text if (text and text.strip()) else "Sorry, I don't have a spoken response."
        safe_text = sanitize_text_for_tts(safe_text)
        if not safe_text:
            safe_text = "Sorry, I don't have a spoken response."
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        tts = gTTS(text=safe_text, lang=language, slow=slow)
        tts.save(out_file_path)
        return out_file_path
    except Exception as e:
        print(f"[voice_service] Error generating TTS: {e}")
        return None
