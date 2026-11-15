"""
Property Prediction Benchmark
==============================
Modular benchmark for molecular property prediction using LLMs.
Supports both API-based models (via LiteLLM) and local models (via vLLM).

Usage (as a script):
    # API model (OpenAI, Anthropic, etc.)
    python -m Property_Prediction.Property_prediction \
        --dataset BBBP --model gpt-4 --backend api \
        --strategy random --k 8 --n 1 --temperature 0.0

    # Local HuggingFace model via vLLM
    python -m Property_Prediction.Property_prediction \
        --dataset Tox21 --model meta-llama/Llama-2-7b-chat-hf --backend local \
        --strategy scaffold --k 8 --n 1 --temperature 0.0

Usage (as a library):
    from Property_Prediction.Property_prediction import run_benchmark
    results = run_benchmark(
        dataset_name="BBBP", model_name="gpt-4", backend="api",
        sample_strategy="random", k=8,
    )
"""

import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm

from .data import load_dataset, LABEL_COL_MAP, TASK_DESCRIPTION
from utils.backend import build_model
from utils.parser import extract_json_answer, strip_thinking
from utils.metrics import compute_property_predictions
from utils.aggregate import (
    pad_outputs, split_runs, aggregate_runs, add_run_columns, format_mean_std,
)


# ── Supported datasets ──────────────────────────────────────────────────────
SUPPORTED_DATASETS = ["BBBP", "BACE", "Tox21", "HIV", "ClinTox"]

def _parse_prediction(raw_text, dataset_name):
    """Extract the first Yes/No answer relevant to the target task.
    Returns 1 (Yes), 0 (No), or -1 (parse failure)."""
    if not raw_text:
        return -1

    text = raw_text.strip()
    text = strip_thinking(text)
    ans = extract_json_answer(text)

    if ans is not None:
        m = re.search(r"\b(yes|no)\b", ans.lower())
        if m:
            return 1 if m.group(1) == "yes" else 0

    task_key = TASK_DESCRIPTION[dataset_name]
    text = raw_text.strip()

    # Try to find "<TaskKey>: Yes/No" pattern first
    pattern = re.compile(
        rf"{re.escape(task_key)}\s*:\s*(Yes|No)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0

    # Fallback: first standalone Yes or No in the text
    fallback = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if fallback:
        return 1 if fallback.group(1).lower() == "yes" else 0

    return -1


# ── Main benchmark runner ────────────────────────────────────────────────────
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
    Run the property-prediction benchmark on one dataset.

    Parameters
    ----------
    dataset_name : str
        One of BBBP, BACE, Tox21, HIV, ClinTox.
    model_name : str
        Model identifier (e.g. "gpt-4", "meta-llama/Llama-2-7b-chat-hf").
    backend : str
        "api" for LiteLLM-based calls, "local" for vLLM-based local inference.
    sample_strategy : str
        "random" or "scaffold" for few-shot example selection.
    k : int
        Number of few-shot examples (0 for zero-shot).
    n : int
        Number of completions to sample per molecule. Each of the n completions
        is scored as its own run over the *same* test set; metrics are reported
        as mean ± std across runs (model variance, not dataset variance).
    temperature : float
        Sampling temperature.
    api_key : str, optional
        API key (for API backends).
    api_base : str, optional
        Custom API base URL (for API backends).
    max_new_tokens : int
        Max tokens to generate (for local models).
    output_dir : str
        Directory to save results.

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
    system_prompts, user_prompts, df_test = load_dataset(
        name=dataset_name, sample_strategy=sample_strategy, k=k
    )

    label_col = LABEL_COL_MAP[dataset_name]
    y_true = df_test[label_col].astype(int).tolist()

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

    print(f"[*] Running inference on {len(user_prompts)} molecules  (n={n} completions each) …")
    for idx in tqdm(range(len(user_prompts)), desc=dataset_name):
        outputs = model.generate(
            user_prompt=user_prompts[idx],
            sys_prompt=system_prompts[idx],
            n=n,
        )
        per_item_outputs.append(pad_outputs(outputs, n))

    # run-major: raw_outputs[r][idx] is run r's completion for item idx
    raw_outputs = split_runs(per_item_outputs, n)
    predictions = [[_parse_prediction(t, dataset_name) for t in run] for run in raw_outputs]

    # ── 4. Evaluate: score every run, then report mean ± std ─────────────
    print("[*] Computing metrics …")
    per_run_metrics = [compute_property_predictions(y_true, preds) for preds in predictions]
    metrics = aggregate_runs(per_run_metrics)

    print(f"\n{'═' * 50}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"{'─' * 50}")
    print(f"  Runs     : n={metrics['n_runs']}")
    print(f"  Accuracy : {format_mean_std(metrics, 'accuracy')}")
    print(f"  F1       : {format_mean_std(metrics, 'f1')}")
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
    results_df["ground_truth"] = y_true
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
        description="ChemLLMBench – Molecular Property Prediction Benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name.",
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
        "--strategy", type=str, default="random", choices=["random", "scaffold"],
        help="Few-shot example sampling strategy.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1,
                        help="Completions per molecule; metrics are reported as mean ± std over n runs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides env vars).")
    parser.add_argument("--api-base", type=str, default=None, help="Custom API base URL.")
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
