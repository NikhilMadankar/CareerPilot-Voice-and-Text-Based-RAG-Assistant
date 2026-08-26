import json
import os
import re
from difflib import get_close_matches

# Path to the extracted courses JSON
DATA_PATH = os.path.join(os.path.dirname(__file__), "rag", "career_courses.json")

_COURSES = []
_COURSE_LOOKUP = {}  # normalized_name -> course dict
_ALL_NAMES_NORMALIZED = []

def _load_courses():
    global _COURSES, _COURSE_LOOKUP, _ALL_NAMES_NORMALIZED
    if _COURSES:
        return
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _COURSES = json.load(f)
        for c in _COURSES:
            norm = _normalize(c.get("course_name", ""))
            _COURSE_LOOKUP[norm] = c
            # Also index aliases or common keywords if helpful
            alias_clean = _normalize(c.get("course_name", "").replace("Engineering", "").replace("Science", "").replace("Sciences", ""))
            if alias_clean and alias_clean not in _COURSE_LOOKUP:
                _COURSE_LOOKUP[alias_clean] = c
        _ALL_NAMES_NORMALIZED = list(_COURSE_LOOKUP.keys())

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())

def find_course(name: str):
    """Find course by exact or fuzzy matching."""
    _load_courses()
    norm = _normalize(name)
    if not norm:
        return None
    if norm in _COURSE_LOOKUP:
        return _COURSE_LOOKUP[norm]

    # Partial / substring match
    for k, v in _COURSE_LOOKUP.items():
        if norm in k or k in norm:
            return v

    # Fuzzy matching
    matches = get_close_matches(norm, _ALL_NAMES_NORMALIZED, n=1, cutoff=0.6)
    if matches:
        return _COURSE_LOOKUP[matches[0]]

    return None

COMPARISON_PATTERNS = [
    r"\b(?:compare|comparison|versus|vs\.?|diff(?:erence)?\s+between)\b",
    r"\b(?:which\s+is\s+better|choose\s+between|better\s+option)\b",
    r"\bor\b.*\bwhich\b",
]

def is_comparison_query(query: str) -> bool:
    """Detect if user query is asking to compare two fields/courses."""
    if not query:
        return False
    q = query.lower()
    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, q):
            return True
    return False

def extract_two_course_names(query: str):
    """
    Attempt to extract two candidate course names from a comparison query.
    e.g. 'Compare Biotechnology with Biomedical Engineering'
         'Biotechnology vs Biomedical Engineering'
         'Difference between Aeronautical Engineering and Aerospace Engineering'
    """
    _load_courses()
    q = query.strip()

    # Clean leading comparison prefixes
    clean_q = re.sub(r"^(?:please\s+)?(?:compare|what is the difference between|difference between|compare between)\s+", "", q, flags=re.IGNORECASE)
    clean_q = re.sub(r"\?+$", "", clean_q)

    delimiters = [r"\s+vs\.?\s+", r"\s+versus\s+", r"\s+and\s+", r"\s+or\s+", r"\s+with\s+", r"\s+to\s+"]
    for delim in delimiters:
        parts = re.split(delim, clean_q, flags=re.IGNORECASE)
        if len(parts) == 2:
            name_a, name_b = parts[0].strip(), parts[1].strip()
            # Clean filler phrases
            name_a = re.sub(r"^(?:which is better|choose between|between)\s+", "", name_a, flags=re.IGNORECASE).strip()
            name_b = re.sub(r"\s+(?:which is better|which one to choose|for me|after 12th)$", "", name_b, flags=re.IGNORECASE).strip()
            
            c_a = find_course(name_a)
            c_b = find_course(name_b)
            if c_a and c_b:
                return c_a["course_name"], c_b["course_name"]

    # Fallback: Check all known courses appearing in the query
    found = []
    norm_q = _normalize(query)
    for c in _COURSES:
        c_norm = _normalize(c["course_name"])
        if c_norm in norm_q and c["course_name"] not in found:
            found.append(c["course_name"])

    if len(found) >= 2:
        return found[0], found[1]

    return None, None

def compare_courses(name_a: str, name_b: str) -> dict:
    """
    Look up both courses from career_courses.json and construct a structured
    comparison prompt for direct LLM completion.
    """
    course_a = find_course(name_a)
    course_b = find_course(name_b)

    if not course_a or not course_b:
        return {
            "ok": False,
            "error": f"Could not match one or both courses: '{name_a}', '{name_b}'",
            "course_a": None,
            "course_b": None,
            "comparison_prompt": ""
        }

    prompt = f"""You are CareerGuide AI, an expert educational counsellor.
Compare the following two academic courses using the official NCERT handbook data provided below:

--- COURSE 1: {course_a['course_name']} ({course_a['category']}, Page {course_a['page_number']}) ---
{course_a['content']}

--- COURSE 2: {course_b['course_name']} ({course_b['category']}, Page {course_b['page_number']}) ---
{course_b['content']}

Please provide a clear, encouraging, and well-structured side-by-side comparison with the following sections:
1. 🎯 Overview & Focus (What each field deals with)
2. 📋 Eligibility & Entrance Exams (10+2 requirements, entrance tests)
3. 🎓 Degree Options (B.Tech, B.Sc, Diploma, PG options)
4. 🏫 Key Institutes / Universities
5. 💡 Career Scope & Guidance (Who should choose which course based on student interests)

Keep the comparison practical, friendly, and well-formatted with bullet points and bold headers.
"""

    return {
        "ok": True,
        "course_a": course_a,
        "course_b": course_b,
        "comparison_prompt": prompt,
        "error": None
    }
