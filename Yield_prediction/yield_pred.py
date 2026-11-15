import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm

# Import your custom data loader
try:
    from .data import load_function
except ImportError:
    from data import load_function
from utils.backend import build_model
from utils.parser import extract_json_answer, strip_thinking
from utils.metrics import compute_yeild_predictions
from utils.aggregate import (
    pad_outputs, split_runs, aggregate_runs, add_run_columns, format_mean_std,
)

SUPPORTED_DATASETS = ["Buchwald-Hartwig", "Suzuki-Miyaura"]

def _parse_prediction(raw_text):
    """Extract the first Yes/No answer relevant to the target task.
    Returns 1 (Yes), 0 (No), or -1 (parse failure)."""
    if not raw_text:
        return -1

    text = raw_text.strip()
    text = strip_thinking(text)
    ans = extract_json_answer(text, key="answer")

    if ans is not None:
        m = re.search(r"\b(yes|no)\b", ans.lower())
        if m:
            return 1 if m.group(1) == "yes" else 0

    # Try to find "yeild: Yes/No" or "yield: Yes/No" pattern first
    pattern = re.compile(
        r"y(?:ei|ie)ld\s*:\s*(Yes|No)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0

    # Fallback: first standalone Yes or No in the text
    fallback = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if fallback:
        return 1 if fallback.group(1).lower() == "yes" else 0

    return -1


def run_benchmark(
    dataset_name: str,
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 50,
    output_dir: str = "results",
):
    """
    Run the yield-prediction benchmark on one dataset.

    n completions are drawn per reaction and scored as n runs over the same
    test set (mean ± std = model variance, not dataset variance).

    Returns
    -------
    dict with keys:
      "metrics"     : mean over the n runs under the usual metric names,
                      "<name>_std" for each numeric metric, "n_runs" and
                      "per_run" (list of the n single-run metric dicts)
      "predictions" : run-major list, predictions[r][i] for run r, item i
      "raw_outputs" : run-major list of raw completions, same layout
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Choose from {SUPPORTED_DATASETS}"
        )

    # ── 1. Load data & prompts ───────────────────────────────────────────
    print(f"[*] Loading dataset: {dataset_name}  (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_function(
        name=dataset_name, sample_strategy=sample_strategy, k=k
    )

    # Map 'Yes' to 1 and 'No' to 0 for metric computation
    y_true = df_test["yeild"].map({"Yes": 1, "No": 0}).tolist()

    # ── 2. Build model ───────────────────────────────────────────────────
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = build_model(
        backend=backend,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # ── 3. Run inference ─────────────────────────────────────────────────
    # The test set and its few-shot prompts were built exactly once above.
    # Every one of the `n` completions of an item reuses that same prompt, so
    # n > 1 measures model (sampling) variance only, never dataset variance.
    per_item_outputs = []  # item-major: per_item_outputs[idx] -> n completions

    print(f"[*] Running inference on {len(user_prompts)} reactions  (n={n} completions each) …")
    for idx in tqdm(range(len(user_prompts)), desc=dataset_name):
        outputs = model.generate(
            user_prompt=user_prompts[idx],
            sys_prompt=system_prompts[idx],
            n=n,
        )
        per_item_outputs.append(pad_outputs(outputs, n))

    # run-major: raw_outputs[r][idx] is run r's completion for item idx
    raw_outputs = split_runs(per_item_outputs, n)
    predictions = [[_parse_prediction(t) for t in run] for run in raw_outputs]

    # ── 4. Evaluate: score every run, then report mean ± std ─────────────
    print("[*] Computing metrics …")
    per_run_metrics = [compute_yeild_predictions(y_true, preds) for preds in predictions]
    metrics = aggregate_runs(per_run_metrics)

    print(f"\n{'═' * 50}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"{'─' * 50}")
    print(f"  Runs     : n={metrics['n_runs']}")
    print(f"  Accuracy : {format_mean_std(metrics, 'accuracy')}")
    print(f"  Valid    : {format_mean_std(metrics, 'valid_predictions', 1)}/{metrics['total']}")
    print(f"{'═' * 50}\n")

    # ── 5. Save results ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"{dataset_name}_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    # Save predictions CSV
    results_df = df_test.copy().reset_index(drop=True)
    add_run_columns(results_df, "raw_output", raw_outputs)
    add_run_columns(results_df, "prediction", predictions)
    add_run_columns(results_df, "prediction_label", [
        ["Yes" if p == 1 else ("No" if p == 0 else "INVALID") for p in preds]
        for preds in predictions
    ])
    results_df["ground_truth_label"] = df_test["yeild"]
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    # Save metrics JSON
    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "backend": backend,
        "sample_strategy": sample_strategy,
        "k": k,
        "n": n,
        "temperature": temperature,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    json_path = os.path.join(output_dir, f"{run_tag}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[✓] Predictions saved to: {csv_path}")
    print(f"[✓] Metrics saved to:     {json_path}")

    return {"metrics": metrics, "predictions": predictions, "raw_outputs": raw_outputs}


# ── CLI entry-point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Reaction Yield Prediction Benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name (Buchwald-Hartwig or Suzuki-Miyaura).",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model identifier (e.g. gpt-4, claude-3-opus-20240229, meta-llama/Llama-2-7b-chat-hf).",
    )
    parser.add_argument(
        "--backend", type=str, default="api", choices=["api", "local"],
        help="'api' for LiteLLM (OpenAI/Anthropic/etc.), 'local' for vLLM.",
    )
    parser.add_argument(
        "--strategy", type=str, default="random", choices=["random"],
        help="Few-shot example sampling strategy.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1,
                        help="Completions per reaction; metrics are reported as mean ± std over n runs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens (local models).")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")

    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
        model_name=args.model,
        backend=args.backend,
        sample_strategy=args.strategy,
        k=args.k,
        n=args.n,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()