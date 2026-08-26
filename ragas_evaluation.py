import json
import os
from ragas_eval import evaluate

if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "ragas_eval.json")
    print(f"🔍 Running evaluation on {data_file}...")
    result = evaluate(data_file)

    if result:
        output_file = os.path.join(os.path.dirname(__file__), "ragas_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Evaluation Completed! Results saved to {output_file}")
