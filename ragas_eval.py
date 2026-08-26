import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(__file__), "ragas_eval.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.60


def to_list_field(x):
    """Normalize input field to list of strings."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(s).strip() for s in x if s and str(s).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(x)]


def maybe_split_paragraphs(text):
    """Split ground-truth block into individual paragraphs if available."""
    if not text or not isinstance(text, str):
        return []
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) > 1:
        return parts
    return [text.strip()]


def evaluate(data_path=None):
    """
    Run semantic RAG evaluation across 4 metrics:
    1. Faithfulness: Is the generated answer grounded in the retrieved context?
    2. Answer Correctness: Semantic similarity against ground-truth answers.
    3. Context Precision: Signal-to-noise ratio of retrieved contexts.
    4. Context Recall: Coverage of ground-truth knowledge in retrieved contexts.
    """
    data_file = data_path or DEFAULT_DATA_FILE
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return {}

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Evaluation dataset is empty.")
        return {}

    model = SentenceTransformer(EMBED_MODEL)
    embed_dim = model.get_sentence_embedding_dimension()

    per_item_results = []
    faithfulness_list = []
    answer_correctness_list = []
    context_precision_list = []
    context_recall_list = []

    for idx, item in enumerate(data):
        question = to_list_field(item.get("question") or item.get("query") or "")
        question = question[0] if question else ""
        answer = to_list_field(item.get("answer") or "")
        answer = answer[0] if answer else ""
        contexts = item.get("contexts") or item.get("context") or item.get("retrieved_chunks")
        contexts = to_list_field(contexts)

        gt_answer = item.get("ground_truth", "") or ""
        gt_contexts_field = item.get("ground_truth_contexts") or item.get("gt_contexts") or None
        if gt_contexts_field:
            gt_contexts = to_list_field(gt_contexts_field)
        else:
            gt_contexts = maybe_split_paragraphs(gt_answer) if gt_answer else []

        def embed_list(texts):
            if not texts:
                return np.zeros((0, embed_dim), dtype=np.float32)
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        a_emb = embed_list([answer])
        q_emb = embed_list([question])
        ctx_embs = embed_list(contexts)
        gt_ans_emb = embed_list([gt_answer])
        gt_ctx_embs = embed_list(gt_contexts)

        def safe_cosine_matrix(A, B):
            if A.size == 0 or B.size == 0:
                return np.zeros((A.shape[0], B.shape[0]))
            return cosine_similarity(A, B)

        # Answer Correctness
        if a_emb.size and gt_ans_emb.size:
            answer_correctness = float(cosine_similarity(a_emb, gt_ans_emb)[0, 0])
        else:
            answer_correctness = float(cosine_similarity(a_emb, q_emb)[0, 0]) if a_emb.size and q_emb.size else 0.0

        # Faithfulness
        if a_emb.size and ctx_embs.size:
            ans_ctx_sims = cosine_similarity(a_emb, ctx_embs)[0]
            faithfulness = float(np.mean(ans_ctx_sims))
        else:
            faithfulness = 0.0

        # Context Precision & Recall
        if ctx_embs.size and gt_ctx_embs.size:
            sim_matrix = safe_cosine_matrix(ctx_embs, gt_ctx_embs)
            matched_per_ctx = (sim_matrix > SIMILARITY_THRESHOLD).any(axis=1)
            context_precision = float(np.sum(matched_per_ctx) / len(contexts)) if len(contexts) > 0 else 0.0
            matched_per_gt = (sim_matrix > SIMILARITY_THRESHOLD).any(axis=0)
            context_recall = float(np.sum(matched_per_gt) / len(gt_contexts)) if len(gt_contexts) > 0 else 0.0
        else:
            context_precision = 0.0
            context_recall = 0.0

        per_item_results.append({
            "index": idx,
            "question": question,
            "answer": answer,
            "n_retrieved_contexts": len(contexts),
            "n_gt_contexts": len(gt_contexts),
            "faithfulness": round(faithfulness, 4),
            "answer_correctness": round(answer_correctness, 4),
            "context_precision": round(context_precision, 4),
            "context_recall": round(context_recall, 4)
        })

        faithfulness_list.append(faithfulness)
        answer_correctness_list.append(answer_correctness)
        context_precision_list.append(context_precision)
        context_recall_list.append(context_recall)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    avg_faith = safe_mean(faithfulness_list)
    avg_acc = safe_mean(answer_correctness_list)
    avg_cprec = safe_mean(context_precision_list)
    avg_crec = safe_mean(context_recall_list)

    summary = {
        "per_item": per_item_results,
        "averages": {
            "faithfulness": round(avg_faith, 4),
            "answer_correctness": round(avg_acc, 4),
            "context_precision": round(avg_cprec, 4),
            "context_recall": round(avg_crec, 4)
        }
    }

    for r in per_item_results:
        print(f"[{r['index']}] Q: {r['question']}")
        print(f"    Faithfulness: {r['faithfulness']:.4f} | Correctness: {r['answer_correctness']:.4f} | "
              f"Precision: {r['context_precision']:.4f} | Recall: {r['context_recall']:.4f}")

    print("\n================== EVALUATION SUMMARY ==================")
    print(f"Average Faithfulness:        {avg_faith:.4f}")
    print(f"Average Answer Correctness:  {avg_acc:.4f}")
    print(f"Average Context Precision:   {avg_cprec:.4f}")
    print(f"Average Context Recall:      {avg_crec:.4f}")
    print("========================================================\n")

    return summary


if __name__ == "__main__":
    evaluate()
