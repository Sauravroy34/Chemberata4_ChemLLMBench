import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm

from utils.backend import build_model
from utils.parser import strip_thinking, extract_json_answer
from utils.metrics import compute_caption_metrices, compute_design_metrices
from utils.aggregate import (
    pad_outputs, split_runs, aggregate_runs, add_run_columns, format_mean_std,
)
from .data import load_dataset, TASK_COL_MAP

# ── Supported tasks ─────────────────────────────────────────────────────────
SUPPORTED_TASKS = list(TASK_COL_MAP.keys())


# ── Parsing ─────────────────────────────────────────────────────────────────
def _parse_prediction(raw_text, task_name):
    if not raw_text:
        return ""

    text = raw_text.strip()
    text = strip_thinking(text)

    ans = extract_json_answer(text)
    if ans:
        return ans.strip()

    
    _, tgt_col = TASK_COL_MAP[task_name]
    match = None

    if task_name == "MOL_CAPTION":
        # captions can be multi-line -> capture everything after the label
        match = re.search(rf"{re.escape(tgt_col)}\s*:\s*(.+)", text,
                          re.IGNORECASE | re.DOTALL)
    elif task_name == "MOL_DESIGN":
        # SMILES contain no whitespace -> first token only
        match = re.search(rf"{re.escape(tgt_col)}\s*:\s*(\S+)", text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fallback: model ignored the label -> first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.split()[0] if task_name == "MOL_DESIGN" else line
    return ""


# ── Metrics (see utils/metrics.py) ──────────────────────────────────────────
def _compute_metrics(y_true, y_pred, task_name):
    """y_true: list of references, y_pred: list of parsed predictions."""
    if task_name == "MOL_CAPTION":
        return compute_caption_metrices(y_true, y_pred)
    elif task_name == "MOL_DESIGN":
        return compute_design_metrices(y_true, y_pred)
    raise ValueError(f"No metrics defined for task: {task_name!r}")


def _print_metrics(metrics, task_name, model_name, sample_strategy, k):
    print(f"\n{'═' * 55}")
    print(f"  Task    : {task_name}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"  Runs    : n={metrics['n_runs']}")
    print(f"{'─' * 55}")
    f = lambda key, prec=4: format_mean_std(metrics, key, prec)
    if task_name == "MOL_CAPTION":
        print(f"  BLEU-2   (↑): {f('bleu2')}")
        print(f"  BLEU-4   (↑): {f('bleu4')}")
        print(f"  ROUGE-1  (↑): {f('rouge1')}")
        print(f"  ROUGE-2  (↑): {f('rouge2')}")
        print(f"  ROUGE-L  (↑): {f('rougeL')}")
        print(f"  METEOR   (↑): {f('meteor')}")
    else:
        print(f"  BLEU        (↑): {f('bleu')}")
        print(f"  Exact       (↑): {f('exact')}")
        print(f"  Levenshtein (↓): {f('levenshtein', 2)}")
        print(f"  Validity    (↑): {f('validity')}")
        print(f"  MACCS FTS   (↑): {f('maccs_fts')}")
        print(f"  RDK FTS     (↑): {f('rdk_fts')}")
        print(f"  Morgan FTS  (↑): {f('morgan_fts')}")
        print(f"  FCD         (↓): {f('fcd')}")
        print(f"  Valid        : {f('valid_predictions', 1)}/{metrics['total']}")
    print(f"{'═' * 55}\n")


# ── Benchmark ───────────────────────────────────────────────────────────────
def run_benchmark(
    task_name: str,
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
):
    """Run the benchmark on one task.

    n completions are drawn per sample and scored as n runs over the same
    test set (mean ± std = model variance, not dataset variance).

    Returns
    dict with keys:
      "metrics"     : mean over the n runs under the usual metric names,
                      "<name>_std" for each numeric metric, "n_runs" and
                      "per_run" (list of the n single-run metric dicts)
      "predictions" : run-major list, predictions[r][i] for run r, item i
      "raw_outputs" : run-major list of raw completions, same layout
    """
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Choose from {SUPPORTED_TASKS}")

    # 1. Load data & prompts
    print(f"[*] Loading task: {task_name}  (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_dataset(
        name=task_name, sample_strategy=sample_strategy, k=k,
    )
    _, to_col = TASK_COL_MAP[task_name]
    y_true = df_test[to_col].astype(str).tolist()

    # 2. Build model
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = build_model(backend, model_name, temperature, max_new_tokens)

    # 3. Inference: n completions per sample, regrouped into n runs
    # The test set and its few-shot prompts were built exactly once above.
    # Every one of the `n` completions of an item reuses that same prompt, so
    # n > 1 measures model (sampling) variance only, never dataset variance.
    per_item_outputs = []  # item-major: per_item_outputs[idx] -> n completions

    print(f"[*] Running inference on {len(user_prompts)} samples  (n={n} completions each) …")
    for idx in tqdm(range(len(user_prompts)), desc=task_name):
        outputs = model.generate(
            user_prompt=user_prompts[idx],
            sys_prompt=system_prompts[idx],
            n=n,
        )
        per_item_outputs.append(pad_outputs(outputs, n))

    # run-major: raw_outputs[r][idx] is run r's completion for item idx
    raw_outputs = split_runs(per_item_outputs, n)
    predictions = [[_parse_prediction(t, task_name) for t in run] for run in raw_outputs]

    # 4. Evaluate every run, then report mean ± std
    print("[*] Computing metrics …")
    per_run_metrics = [_compute_metrics(y_true, preds, task_name) for preds in predictions]
    metrics = aggregate_runs(per_run_metrics)
    _print_metrics(metrics, task_name, model_name, sample_strategy, k)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"{task_name}_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    results_df = df_test.copy().reset_index(drop=True)
    add_run_columns(results_df, "raw_output", raw_outputs)
    add_run_columns(results_df, "prediction", predictions)
    results_df["ground_truth"] = y_true
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    meta = {
        "task": task_name,
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
        description="ChemLLMBench – Molecule Captioning / Design Benchmark"
    )
    parser.add_argument("--task", type=str, required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier (e.g. gpt-4, meta-llama/Llama-2-7b-chat-hf).")
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "scaffold"])
    parser.add_argument("--k", type=int, default=8, help="Few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1,
                        help="Completions per sample; metrics are reported as mean ± std over n runs.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(
        task_name=args.task,
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