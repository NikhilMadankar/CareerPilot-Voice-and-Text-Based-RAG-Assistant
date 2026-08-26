import json
import os
import sys
import urllib.request

# Ensure UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Curated benchmark test cases with golden ground-truth facts from NCERT Compendium
BENCHMARK_QUESTIONS = [
    {
        "question": "What is the eligibility criteria for Aeronautical Engineering after 12th?",
        "ground_truth": "10+2 examination with Physics, Mathematics, and Chemistry (PCM) is required for undergraduate B.Tech/B.E. in Aeronautical Engineering.",
        "ground_truth_contexts": [
            "A course in Aeronautical Engineering includes the designing, manufacturing, testing and maintenance of aircraft in commercial aviation and defence sectors. Under graduate Level: 10+2 examination with Physics, Mathematics, Chemistry."
        ]
    },
    {
        "question": "Compare Biotechnology Engineering vs Biomedical Engineering",
        "ground_truth": "Biotechnology Engineering combines technology with biology for research in genetics, pharmaceuticals, and agriculture. Biomedical Engineering applies engineering to medicine, prosthetics, and medical diagnostic equipment. Both require 10+2 with Science subjects (Biology/Maths/Chemistry).",
        "ground_truth_contexts": [
            "Biotechnology engineering is a branch of engineering where technology is combined with biology for research and development.",
            "Biomedical engineering is the study of engineering as applied in the medical sector such as manufacturing prostheses, medical equipment, diagnostic devices and drugs."
        ]
    },
    {
        "question": "What are the courses and career prospects in Artificial Intelligence and Machine Learning?",
        "ground_truth": "Eligibility is 10+2 with Science stream. Courses include B.Tech Computer Science with specialization in AI & Machine Learning, Advanced Certification in AI/ML, and M.Tech in AI or Robotics. Notable institutes include IISc Bangalore, IIT Bombay, IIT Kharagpur, and IIIT Hyderabad.",
        "ground_truth_contexts": [
            "Artificial Intelligence belongs to a field of science and engineering in which studies and research aim to develop intelligent computer machines that can perform tasks with human intelligence. Eligibility: 10+2 Science. Courses: B. Tech Computer Science & Engineering with specialization in AI & Machine Learning, Master of Technology in Artificial Intelligence. Institutes: IISc Bangalore, IIT Bombay, IIT Kharagpur, IIIT Hyderabad, IIT Madras."
        ]
    },
    {
        "question": "What degree options and institutes are available for Architecture Engineering?",
        "ground_truth": "Degree options include B.Arch, M.Arch, and Ph.D. Programmes. Eligibility is 10+2 level with Science Stream and JEE scores. Notable institutes include IIT Kharagpur, IIT Roorkee, CEPT University Ahmedabad, and School of Planning and Architecture Delhi.",
        "ground_truth_contexts": [
            "Architecture is the science that deals with planning, designing, safety, affordability, and supervision of construction works. Eligibility: 10+2 level with Science Stream. Courses: B. Arch, M. Arch, Ph. D Programmes. Institutes: IIT Kharagpur, IIT Roorkee, CEPT University, School of Planning and Architecture Delhi, NIT Patna."
        ]
    },
    {
        "question": "Tell me about career options and eligibility in Civil Engineering.",
        "ground_truth": "Civil Engineering involves planning, designing, and constructing structural works like roads, bridges, buildings, and dams. Eligibility requires 10+2 with Physics, Chemistry, and Mathematics. Degrees include B.Tech, M.Tech, and Ph.D.",
        "ground_truth_contexts": [
            "Civil Engineering involves planning, designing and executing structural works including roads, bridges, tunnels, buildings, airports, dams, water works. Eligibility: 10+2 with Physics, Chemistry, and Mathematics as core subjects. Courses: B. Tech, M. Tech (Dual Degree), Ph. D."
        ]
    },
    {
        "question": "What are the entrance exams and career pathway for Astronomy and Astrophysics?",
        "ground_truth": "Eligibility requires 10+2 with PCM. For higher studies/PhD, entrance exams include INAT (IUCAA-NCRA Admission Test), JEST, and CSIR-UGC NET for JRF. Notable institutes include Indian Institute of Astrophysics Bangalore, IISc Bangalore, and IUCAA Pune.",
        "ground_truth_contexts": [
            "Astronomy is a combination of physics, chemistry and mathematical principles. Astrophysics explores properties of astronomical objects. Eligibility: 10 +2 with PCM. Entrance Tests: INAT, JEST, CSIR-UGC NET for JRF. Institutes: Indian Institute of Astrophysics, IISc Bangalore, IUCAA Pune, ARIES Nainital."
        ]
    },
    {
        "question": "What is the difference between Aeronautical Engineering and Aerospace Engineering?",
        "ground_truth": "Aeronautical Engineering deals specifically with aircraft operating within the Earth's atmosphere, whereas Aerospace Engineering is broader and covers both aeronautical craft in Earth's atmosphere and astronautical craft/spacecraft operating in outer space.",
        "ground_truth_contexts": [
            "A course in Aeronautical Engineering includes the designing, manufacturing, testing and maintenance of aircraft in commercial aviation and defence sectors.",
            "Aerospace engineering is divided into two major branches: aeronautical engineering related with aircrafts in the earth's atmosphere, and astronautical engineering that deals with spacecrafts that operate outside the earth's atmosphere."
        ]
    },
    {
        "question": "What courses are available in Ceramics Engineering and what are the top institutes?",
        "ground_truth": "Ceramics Engineering involves creating heat-resistant objects from inorganic, non-metallic materials for mining, aerospace, electronics, and medicine. Courses include B.Tech and M.Tech. Notable institutes include IITs, Government College of Engineering and Ceramic Technology Kolkata, and Andhra University.",
        "ground_truth_contexts": [
            "Ceramic engineering is the science and technology of creating objects from inorganic, non-metallic materials. Used in mining, aerospace, medicine, electronics. Courses: B. Tech, M. Tech. Institutes: IITs, Government College of Engineering and Ceramic Technology Kolkata, Andhra University, RTU Kota."
        ]
    }
]


def query_pipeline(query: str):
    """Query live server endpoint or fallback to direct RAG import."""
    url = "http://127.0.0.1:5000/chat"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("answer", ""), data.get("retrieved_chunks", [])
    except Exception as e:
        print(f"  (HTTP query fallback: {e})")
        from app import get_career_response
        result = get_career_response(query)
        return result.get("answer", ""), result.get("retrieved_chunks", [])


def generate_evaluation_dataset(output_file="ragas_eval.json"):
    print("🚀 Generating evaluation dataset by querying CareerGuide AI benchmark...")
    dataset = []

    for idx, item in enumerate(BENCHMARK_QUESTIONS, 1):
        q = item["question"]
        print(f"[{idx}/{len(BENCHMARK_QUESTIONS)}] Querying: {q}")
        
        answer, contexts = query_pipeline(q)
        
        dataset.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "ground_truth_contexts": item["ground_truth_contexts"]
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Benchmark dataset with {len(dataset)} items saved to '{output_file}' successfully!")


if __name__ == "__main__":
    generate_evaluation_dataset()
