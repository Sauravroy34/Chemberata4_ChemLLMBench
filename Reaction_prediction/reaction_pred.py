import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger

from utils.backend import build_model
from utils.parser import extract_json_answer, strip_thinking
from utils.metrics import compute_reaction_predictions, _is_invalid_prediction
from utils.aggregate import (
    pad_outputs, split_runs, aggregate_runs, add_run_columns, format_mean_std,
)
# Adjust the module path / function name to match your forward-reaction data file
from .data import (
    load_forward_dataset as load_dataset,
    INPUT_COL, TARGET_COL, INPUT_LABEL, TARGET_LABEL,
)

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings for invalid SMILES

TASK_NAME = "FORWARD_REACTION"


# ── Parsing ─────────────────────────────────────────────────────────────────
def _clean(text):
    """Strip markdown fences, quotes and backticks."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.strip().strip("`'\" \n")


def _parse_prediction(raw_text):
    """Extract the predicted product SMILES (a single whitespace-free token)."""
    if not raw_text:
        return ""
    text = _clean(raw_text)
    text = strip_thinking(text)
    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()


    # Preferred: text after the target label, e.g. "Products: CC(=O)OCC".
    # Tolerates "Product:" as well as "Products:".
    label = re.escape(TARGET_LABEL.rstrip("sS"))
    match = re.search(rf"{label}s?\s*:\s*(\S+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip("`'\"")

    # Fallback: model ignored the label -> first token of first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.split()[0].strip("`'\"")
    return ""


# ── Metrics (see utils/metrics.py) ──────────────────────────────────────────
def _print_metrics(metrics, model_name, sample_strategy, k):
    print(f"\n{'═' * 55}")
    print("  Task    : FORWARD REACTION PREDICTION (top-1)")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"  Runs    : n={metrics['n_runs']}")
    print(f"{'─' * 55}")
    print(f"  Top-1 Accuracy       (↑): {format_mean_std(metrics, 'top1_accuracy')}"
          f"  ({format_mean_std(metrics, 'correct', 1)}/{metrics['total']})")
    print(f"  Invalid SMILES (%)   (↓): {format_mean_std(metrics, 'invalid_smiles_percent', 2)}"
          f"  ({format_mean_std(metrics, 'invalid', 1)}/{metrics['total']})")
    print(f"{'═' * 55}\n")


# ── Benchmark ───────────────────────────────────────────────────────────────
def run_benchmark(
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
):
    """Run top-1 forward reaction prediction benchmark.

    n completions are drawn per reaction and scored as n runs over the same
    test set (mean ± std = model variance, not dataset variance).

    Returns
    dict with keys:
      "metrics"     : mean over the n runs under the usual metric names,
                      "<name>_std" for each numeric metric, "n_runs" and
                      "per_run" (list of the n single-run metric dicts)
      "predictions" : run-major list, predictions[r][i] for run r, item i
      "raw_outputs" : run-major list of raw completions, same layout
    """
    # 1. Load data & prompts
    print(f"[*] Loading forward reaction data (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_dataset(
        sample_strategy=sample_strategy, k=k,
    )
    y_true = df_test[TARGET_COL].astype(str).tolist()

    # 2. Build model
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = build_model(backend, model_name, temperature, max_new_tokens)

    # 3. Inference: n completions per reaction, regrouped into n runs
    # The test set and its few-shot prompts were built exactly once above.
    # Every one of the `n` completions of an item reuses that same prompt, so
    # n > 1 measures model (sampling) variance only, never dataset variance.
    per_item_outputs = []  # item-major: per_item_outputs[idx] -> n completions

    print(f"[*] Running inference on {len(user_prompts)} samples  (n={n} completions each) …")
    for idx in tqdm(range(len(user_prompts)), desc="FORWARD"):
        outputs = model.generate(
            user_prompt=user_prompts[idx],
            sys_prompt=system_prompts[idx],
            n=n,
        )
        per_item_outputs.append(pad_outputs(outputs, n))

    # run-major: raw_outputs[r][idx] is run r's completion for item idx
    raw_outputs = split_runs(per_item_outputs, n)
    predictions = [[_parse_prediction(t) for t in run] for run in raw_outputs]

    # 4. Evaluate every run, then report mean ± std
    print("[*] Computing metrics …")
    per_run = [compute_reaction_predictions(y_true, preds) for preds in predictions]
    per_run_metrics = [m for m, _ in per_run]
    correct_flags = [flags for _, flags in per_run]  # run-major
    metrics = aggregate_runs(per_run_metrics)
    _print_metrics(metrics, model_name, sample_strategy, k)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"FORWARD_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    results_df = df_test.copy().reset_index(drop=True)
    results_df["ground_truth"] = y_true
    add_run_columns(results_df, "raw_output", raw_outputs)
    add_run_columns(results_df, "prediction", predictions)
    add_run_columns(results_df, "is_correct", correct_flags)
    add_run_columns(results_df, "is_invalid",
                    [[_is_invalid_prediction(p) for p in preds] for preds in predictions])
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    meta = {
        "task": TASK_NAME,
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


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Forward Reaction Prediction (top-1) Benchmark"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier (e.g. gpt-4, meta-llama/Llama-2-7b-chat-hf).")
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "scaffold"])
    parser.add_argument("--k", type=int, default=8, help="Few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1,
                        help="Completions per reaction; metrics are reported as mean ± std over n runs.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(
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