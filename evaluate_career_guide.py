import os
import sys
import json
from ragas_eval import evaluate

# Ensure UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_career_guide_evaluation(data_file="ragas_eval.json", output_file="ragas_results.json"):
    print(f"📊 Running Semantic RAG Evaluation on '{data_file}'...")
    
    if not os.path.exists(data_file):
        print(f"⚠️ Error: Evaluation data file '{data_file}' not found. Please run 'python generate_eval_dataset.py' first.")
        return

    summary = evaluate(data_file)

    if summary and "averages" in summary:
        avg = summary["averages"]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n========================================================")
        print("          🎓 CAREER GUIDE AI - EVALUATION METRICS       ")
        print("========================================================")
        print(f"  • Faithfulness:         {avg['faithfulness'] * 100:.1f}%")
        print(f"  • Answer Correctness:   {avg['answer_correctness'] * 100:.1f}%")
        print(f"  • Context Precision:    {avg['context_precision'] * 100:.1f}%")
        print(f"  • Context Recall:       {avg['context_recall'] * 100:.1f}%")
        print("========================================================")
        print(f"Detailed per-item results saved to '{output_file}'\n")

    return summary


if __name__ == "__main__":
    run_career_guide_evaluation()
