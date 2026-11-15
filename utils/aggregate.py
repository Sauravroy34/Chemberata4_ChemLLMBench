
import math
import numbers


# ── Regrouping completions ──────────────────────────────────────────────────
def pad_outputs(outputs, n):
    """Return exactly ``n`` strings: truncate extras, pad shortfalls with ''."""
    outputs = list(outputs or [])
    outputs = ["" if o is None else str(o) for o in outputs]
    if len(outputs) < n:
        outputs.extend([""] * (n - len(outputs)))
    return outputs[:n]


def split_runs(per_item_outputs, n):
    """Item-major -> run-major: ``result[r][i] == per_item_outputs[i][r]``."""
    return [[item[r] for item in per_item_outputs] for r in range(n)]


def add_run_columns(df, name, per_run_values):
    """Write one column per run to ``df``: ``<name>_run0``, ``<name>_run1``, ..."""
    for r, values in enumerate(per_run_values):
        df[f"{name}_run{r}"] = list(values)
    return df


# ── Statistics ──────────────────────────────────────────────────────────────
def _is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _to_plain(v):
    """numpy scalars -> builtin int / float so the JSON dump stays clean."""
    if isinstance(v, numbers.Integral):
        return int(v)
    return float(v)


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    """Sample standard deviation (ddof=1). 0.0 for a single value. NaN propagates."""
    k = len(xs)
    if k < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (k - 1))


def aggregate_runs(per_run_metrics):
    """
    Reduce a list of per-run metric dicts to one summary dict.

    The result keeps the *same keys* as a single-run metrics dict so existing
    consumers (``metrics["accuracy"]`` etc.) keep working:

      * numeric key   -> mean across runs
      * ``<key>_std`` -> sample std (ddof=1) across runs; 0.0 when n_runs == 1
      * non-numeric key (e.g. ``"task"``) -> value from the first run
      * ``"n_runs"``  -> number of runs
      * ``"per_run"`` -> the untouched list of per-run metric dicts

    A numeric value that is identical in every run (e.g. ``"total"``) is kept
    as-is (int stays int) instead of being turned into a float mean.
    """
    if not per_run_metrics:
        raise ValueError("aggregate_runs() needs at least one run")

    n_runs = len(per_run_metrics)
    keys = []
    for m in per_run_metrics:
        for k in m:
            if k not in keys:
                keys.append(k)

    out = {}
    for k in keys:
        values = [m[k] for m in per_run_metrics if k in m]
        if values and all(_is_number(v) for v in values):
            plain = [_to_plain(v) for v in values]
            if all(v == plain[0] for v in plain):
                out[k] = plain[0]
            else:
                out[k] = _mean(plain)
            out[f"{k}_std"] = _std(plain)
        else:
            out[k] = values[0]

    out["n_runs"] = n_runs
    out["per_run"] = per_run_metrics
    return out


def format_mean_std(metrics, key, prec=4):
    """``'0.4200 ± 0.0141'`` when more than one run, else just ``'0.4200'``."""
    mean = metrics[key]
    std = metrics.get(f"{key}_std")
    text = f"{mean:.{prec}f}"
    if metrics.get("n_runs", 1) > 1 and std is not None:
        text += f" ± {std:.{prec}f}"
    return text
