import os
import re
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger
from pathlib import Path

from utils.backend import build_model
from utils.parser import extract_json_answer, strip_thinking
from utils.metrics import compute_reaction_selection
from utils.aggregate import (
    pad_outputs, split_runs, aggregate_runs, add_run_columns, format_mean_std,
)
from .data import load_dataset, TASK_MAP as CSV_MAP   # task -> csv filename

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

TASKS = ["Reactant_selection", "Ligand_selection", "Solvent_selection"]

# Label the model is asked to answer with (used for parsing)
LABEL_MAP = {
    "Reactant_selection": "Optimal reactant",
    "Ligand_selection": "Optimal ligand",
    "Solvent_selection": "Optimal solvent",
}

# Any component label a model might prefix its answer with
_COMPONENT = r"(?:reactant|reagent|ligand|solvent|base)s?"


# ── Parsing ─────────────────────────────────────────────────────────────────
def _clean(text):
    """Strip markdown fences, bold markers, quotes and backticks."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("**", "")
    return text.strip().strip("`'\" \n")


def _strip_token(tok):
    """Trim wrapping quotes/backticks and trailing punctuation from a token."""
    tok = tok.strip("`'\",;")
    # a SMILES never ends with '.', so a trailing period is sentence punctuation
    return tok.rstrip(".").strip("`'\",;")


def _parse_prediction(raw_text, task):
    """Extract the predicted SMILES string from the model's raw output.

    Handles, e.g.:
        "Optimal reactant: Brc1ccc2ncccc2c1"
        "Reactant: Brc1ccc2ncccc2c1"        (label differs from the template)
        "Reactant:Brc1ccc2ncccc2c1"         (no space after the colon)
        "The optimal solvent is: CO."       (sentence form)
        "Brc1ccc2ncccc2c1"                  (no label at all)
        "Reactant: X\\nExplanation ..."      (extra text after the answer)
    """
    if not raw_text:
        return ""
    text = _clean(str(raw_text))
    if not text:
        return ""

    text = strip_thinking(text)
    ans = extract_json_answer(text, key="answer")

    if ans:
        return ans.strip()

    # 1) Task-specific label, e.g. "Optimal solvent: CO"
    label = re.escape(LABEL_MAP[task].strip())
    m = re.search(rf"{label}s?\s*(?:is\s*)?[:=]\s*(\S+)", text, re.IGNORECASE)
    if m:
        return _strip_token(m.group(1))

    # 2) Generic "Optimal <anything>: X"
    m = re.search(r"optimal\s+\w+\s*(?:is\s*)?[:=]\s*(\S+)", text, re.IGNORECASE)
    if m:
        return _strip_token(m.group(1))

    # 3) Any component label at the start of a line: "Reactant: X", "Solvent:X"
    m = re.search(rf"^\s*{_COMPONENT}\s*(?:is\s*)?[:=]\s*(\S+)",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        return _strip_token(m.group(1))

    # 4) No label: first token of the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _strip_token(line.split()[0])
    return ""


# ── Metrics (see utils/metrics.py) ──────────────────────────────────────────
def _print_metrics(task, metrics, model_name):
    print(f"\n{'═' * 55}")
    print(f"  Task    : {task} (zero-shot)")
    print(f"  Model   : {model_name}")
    print(f"  Runs    : n={metrics['n_runs']}")
    print(f"{'─' * 55}")
    print(f"  Accuracy (↑): {format_mean_std(metrics, 'accuracy')}"
          f"  ({format_mean_std(metrics, 'correct', 1)}/{metrics['total']})")
    print(f"{'═' * 55}\n")


# ── Benchmark ───────────────────────────────────────────────────────────────
def run_task(task, model, model_name, data_dir, n=1):

    csv_path = str(Path(__file__).parent/CSV_MAP[task])
    print(f"[*] Loading {task} from {csv_path}")
    system_prompts, user_prompts, df = load_dataset(csv_path)
    df = df.reset_index(drop=True)
    y_true = df["ground_truth"].astype(str).tolist()

    candidate_rank = df["candidate_rank"].to_list()

    # The test set and its few-shot prompts were built exactly once above.
    # Every one of the `n` completions of an item reuses that same prompt, so
    # n > 1 measures model (sampling) variance only, never dataset variance.
    per_item_outputs = []  # item-major: per_item_outputs[idx] -> n completions

    print(f"[*] Running zero-shot inference on {len(user_prompts)} samples  (n={n} completions each) …")
    for idx in tqdm(range(len(user_prompts)), desc=task):
        outputs = model.generate(
            user_prompt=user_prompts[idx],
            sys_prompt=system_prompts[idx],
            n=n,
        )
        per_item_outputs.append(pad_outputs(outputs, n))

    # run-major: raw_outputs[r][idx] is run r's completion for item idx
    raw_outputs = split_runs(per_item_outputs, n)
    predictions = [[_parse_prediction(t, task) for t in run] for run in raw_outputs]

    per_run = [compute_reaction_selection(y_true, preds, candidate_rank, task)
               for preds in predictions]
    per_run_metrics = [m for m, _ in per_run]
    correct_flags = [flags for _, flags in per_run]  # run-major
    metrics = aggregate_runs(per_run_metrics)
    _print_metrics(task, metrics, model_name)

    results_df = df.copy()
    add_run_columns(results_df, "raw_output", raw_outputs)
    add_run_columns(results_df, "prediction", predictions)
    add_run_columns(results_df, "is_correct", correct_flags)
    return metrics, results_df, raw_outputs, predictions


def run_benchmark(
    model_name: str,
    backend: str = "api",
    tasks=None,
    data_dir: str = ".",
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    k: int = 5,
    sample_strategy: str = "random",
    output_dir: str = "results",
):
    """Zero-shot reagent / ligand / solvent selection benchmark (accuracy only).

    n completions are drawn per sample and scored as n runs over the same
    test set (mean ± std = model variance, not dataset variance).

    Returns dict: task -> {"metrics", "predictions", "raw_outputs"} where
    "metrics" holds mean / "<name>_std" / "n_runs" / "per_run" and the other
    two are run-major lists ([run][item]).
    """
    tasks = tasks or TASKS
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = build_model(backend, model_name, temperature, max_new_tokens)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")

    all_metrics, results = {}, {}
    for task in tasks:
        metrics, results_df, raw_outputs, predictions = run_task(
            task, model, model_name, data_dir, n=n
        )
        all_metrics[task] = metrics
        results[task] = {"metrics": metrics, "predictions": predictions,
                         "raw_outputs": raw_outputs}

        run_tag = f"{task}_{safe_model}_zeroshot_{timestamp}"
        csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"[✓] {task} predictions saved to: {csv_path}")

    # Summary
    macro_acc = sum(m["accuracy"] for m in all_metrics.values()) / len(all_metrics)
    print(f"\n{'═' * 55}\n  SUMMARY | {model_name} (zero-shot)\n{'─' * 55}")
    for t, m in all_metrics.items():
        print(f"  {t:<20} {format_mean_std(m, 'accuracy')}  ({format_mean_std(m, 'correct', 1)}/{m['total']})")
    print(f"  {'MACRO_AVG':<20} {macro_acc:.4f}\n{'═' * 55}\n")

    meta = {
        "model": model_name,
        "backend": backend,
        "setting": "zero-shot",
        "n": n,
        "temperature": temperature,
        "timestamp": timestamp,
        "metrics": all_metrics,
        "macro_avg_accuracy": macro_acc,
    }
    json_path = os.path.join(output_dir, f"SELECTION_{safe_model}_zeroshot_{timestamp}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[✓] Metrics saved to: {json_path}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Reagent / Ligand / Solvent selection (zero-shot) accuracy benchmark"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing the *_sample.csv files.")
    parser.add_argument("--n", type=int, default=1,
                        help="Completions per sample; metrics are reported as mean ± std over n runs.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        backend=args.backend,
        tasks=args.tasks,
        data_dir=args.data_dir,
        n=args.n,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()