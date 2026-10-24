import os
import pickle
from pathlib import Path
import random
import requests
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold



REACTION_PREDICTION_PROMPT = """
You are an expert chemist. Given the reactants SMILES, your task is to predict the main product SMILES using your experienced chemical Reaction Prediction knowledge. 
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the product. The input contains the reactants and reagents which are split by '.'. The product smiles must be valid and chemically reasonable. 
"""

# ----------------------------- config -----------------------------
SEED = 42
SAMPLE_SIZE = 100
_DATA_DIR = Path(__file__).parent
TEST_SET_CSV = str(_DATA_DIR / "uspto_test.csv")   # if this file exists it is used as the test set

random.seed(SEED)

SHARED_NAME = "7eci3nd9vy0xplqniitpk02rbg9q2zcq"
FILES = {"forward": "854847816719"}

# the pickle holds *_mol columns; they are renamed first and converted to SMILES lazily


MAPPING_COL_NAMES = {
    "reactants_mol": "reactant",
    "products_mol": "product",
}

INPUT_COL = "reactant"     # what the model is given
TARGET_COL = "product"   # what the model must predict


INPUT_LABEL = "reactant"
TARGET_LABEL = "Products"
_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ----------------------------- bare-bytes unpickling -----------------------------
class MolStub:
    """Holds the raw RDKit binary instead of building a full Mol."""
    def __init__(self, *args):
        self.b = args[0] if args else None

    def __setstate__(self, state):
        pass  # ignore any extra state


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("rdkit") and name == "Mol":
            return MolStub
        return super().find_class(module, name)


# ----------------------------- download / load -----------------------------
def download_box_file(file_id, out_path):
    """Download a file from a public Box shared link (atomic, cached)."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Using cached file: {out_path}")
        return out_path

    url = "https://az.app.box.com/index.php"
    params = {
        "rm": "box_download_shared_file",
        "shared_name": SHARED_NAME,
        "file_id": f"f_{file_id}",
    }
    tmp_path = out_path + ".part"

    with requests.get(url, params=params, stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError("Got an HTML page instead of the file. "
                               "Download it manually from the browser instead.")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    os.replace(tmp_path, out_path)
    print(f"Downloaded: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def load_raw_dataset(name):
    """Loads the DataFrame with MolStub objects (raw bytes) instead of real Mols."""
    path = download_box_file(FILES[name], f"{name}.pkl")
    with open(path, "rb") as f:
        obj = StubUnpickler(f).load()
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


# ----------------------------- preprocessing (in place) -----------------------------
def _safe_smiles(m):
    """MolStub (raw bytes) -> canonical SMILES. One real Mol alive at a time."""
    try:
        return Chem.MolToSmiles(Chem.Mol(m.b))
    except Exception:
        return None


def convert_stub_cols(df, cols=(INPUT_COL, TARGET_COL)):
    """Stub (raw bytes) -> canonical SMILES, in place, for the given df only."""
    for col in cols:
        df[col] = df[col].map(_safe_smiles)


def drop_columns(df, cols):
    df.drop(columns=list(cols), inplace=True, errors="ignore")


def rename_columns(df, mapping):
    df.rename(columns=mapping, inplace=True)


def filter_invalid_smiles(df, cols=(INPUT_COL, TARGET_COL)):
    """Drop rows where any of the given SMILES columns is missing/invalid (in place).
    Call only after the columns have been converted to SMILES."""
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        mask &= np.fromiter(
            (isinstance(s, str) and Chem.MolFromSmiles(s) is not None for s in df[col]),
            dtype=bool, count=len(df),
        )
    df.drop(index=df.index[~mask], inplace=True)
    df.reset_index(drop=True, inplace=True)


def split_train_test(df, split_col="set", train_label="train", test_label="test",
                     drop_split_col=True):
    cols = [c for c in df.columns if not (drop_split_col and c == split_col)]
    labels = df[split_col]
    train_df = df.loc[labels == train_label, cols].reset_index(drop=True)
    test_df = df.loc[labels == test_label, cols].reset_index(drop=True)
    return train_df, test_df


def load_or_sample_test_set(test_df, csv_path=TEST_SET_CSV, n=SAMPLE_SIZE, seed=SEED):
    """Use the original test-set CSV if present, otherwise sample n rows (not saved).
    Sampling happens first; only the sampled rows are converted to SMILES."""
    if os.path.exists(csv_path):
        print(f"Using existing test set: {csv_path}")
        return pd.read_csv(csv_path)          # already SMILES

    n = min(n, len(test_df))
    print(f"No test set CSV found, sampling {n} rows randomly (seed={seed})")
    df_test = test_df.sample(n=n, random_state=seed).reset_index(drop=True)
    convert_stub_cols(df_test)                # only n rows converted
    return df_test


# ----------------------------- ICL example sampling -----------------------------
def _format_examples(rows):
    """List of (input_smiles, target_smiles) tuples."""
    return list(zip(rows[INPUT_COL], rows[TARGET_COL]))


def _random_sample_examples(df_train, k):
    """k random examples, shared by every test molecule. Only these k rows are converted."""
    rows = df_train.sample(k, random_state=SEED).copy()
    convert_stub_cols(rows)
    return _format_examples(rows)


def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
    return _fp_gen.GetFingerprint(scaffold)


def _scaffold_sample_examples(df_train, train_fps, test_smiles, k):
    """k train examples whose scaffolds are most similar to the test molecule."""
    test_fp = _scaffold_fingerprint(test_smiles)
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(test_fp, train_fps))
    top_k = np.argsort(sims)[::-1][:k]
    return _format_examples(df_train.iloc[top_k])


# ----------------------------- prompt building -----------------------------
def _build_system_prompt(examples):
    """Base prompt plus an examples block."""
    system_prompt = REACTION_PREDICTION_PROMPT
    if not examples:
        return system_prompt
    system_prompt += "\n" + "\n\n".join(
        f"{INPUT_LABEL}: {inp}\n{TARGET_LABEL}: {tgt}" for inp, tgt in examples
    )
    return system_prompt


def _create_prompt(sample_strategy, df_train, df_test, k):
    """Returns (system_prompts, user_prompts), one entry per test reaction."""
    test_smiles = df_test[INPUT_COL].tolist()
    user_prompts = [f"{INPUT_LABEL}: {s}\n{TARGET_LABEL}:" for s in test_smiles]

    # zero-shot: train_df is never touched, nothing converted
    if k == 0:
        return [_build_system_prompt(None)] * len(test_smiles), user_prompts

    # random: sample first, convert just k rows
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(_random_sample_examples(df_train, k))
        return [system_prompt] * len(test_smiles), user_prompts

    # scaffold: needs every train molecule, so convert the whole train set here
    if sample_strategy == "scaffold":
        convert_stub_cols(df_train)           # the only place the full train set is converted
        train_fps = [_scaffold_fingerprint(s) for s in df_train[INPUT_COL]]  # computed once
        system_prompts = [
            _build_system_prompt(_scaffold_sample_examples(df_train, train_fps, s, k))
            for s in test_smiles
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")


# ----------------------------- main entry point -----------------------------
def load_forward_dataset(sample_strategy="random", k=5, name="forward"):
    """Returns (system_prompts, user_prompts, df_test)."""

    df = load_raw_dataset(name)               # stubs, cheap
    print("this is done")
    drop_columns(df, ["reaction_type"])
    rename_columns(df, MAPPING_COL_NAMES)
    train_df, test_df = split_train_test(df)
    df_test = load_or_sample_test_set(test_df)
    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy, df_train=train_df, df_test=df_test, k=k
    )
    return system_prompts, user_prompts, df_test



if __name__ == "__main__":

    sys_p, usr_p, df = load_forward_dataset(k=0)

    print(sys_p[0])
    print(usr_p[0])
