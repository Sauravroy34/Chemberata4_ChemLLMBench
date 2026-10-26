MOL_CAPTION = """
You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n\
Please strictly follow the format, no other information can be provided. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""

MOL_DESIGN = """
"You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES \
string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""











import pandas as pd
import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

FILES = {
    "train": "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data/train.txt",
    "test": "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data/test.txt"
}

TASK_COL_MAP = {
    "MOL_CAPTION": ("smiles", "description"),
    "MOL_DESIGN":  ("description", "smiles"),
}

PROMPTS = {
    "MOL_CAPTION": MOL_CAPTION,
    "MOL_DESIGN":  MOL_DESIGN,
}

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _canonicalise_smiles(df):
    """Convert all SMILES to canonical form, dropping invalid entries."""
    df.dropna(subset=["smiles"], inplace=True)
    
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df.dropna(subset=["mol"], inplace=True)
    
    df["smiles"] = df["mol"].apply(Chem.MolToSmiles)
    df.drop(columns=["mol"], inplace=True)
    
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Example formatting
# ---------------------------------------------------------------------------

def _format_examples(rows, name):
    """Return a list of (input_value, output_value) tuples for the given task."""
    from_col, to_col = TASK_COL_MAP[name]
    return list(zip(rows[from_col], rows[to_col]))


def _random_sample_examples(df, k, name):
    """Randomly sample *k* training examples for in-context learning."""
    sampled = df.sample(k, random_state=SEED)
    return _format_examples(sampled, name)

def _scaffold_fingerprint(smiles):
    """Morgan fingerprint of the Murcko scaffold of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return _fp_gen.GetFingerprint(scaffold)

def _scaffold_sample_examples(df_train, train_fps, test_smiles, k, name):
    """Select the *k* training molecules whose scaffolds are most
    Tanimoto-similar to the test molecule's scaffold."""
    test_fp = _scaffold_fingerprint(test_smiles)
    
    # Filter out any None fingerprints in case of scaffold errors
    valid_indices = [i for i, fp in enumerate(train_fps) if fp is not None]
    valid_fps = [train_fps[i] for i in valid_indices]
    
    if test_fp is None or not valid_fps:
        # Fallback to random if scaffold extraction fails
        return _random_sample_examples(df_train, k, name)
        
    similarities = DataStructs.BulkTanimotoSimilarity(test_fp, valid_fps)
    top_k_valid_idx = np.argsort(similarities)[::-1][:k]
    top_k_original_idx = [valid_indices[i] for i in top_k_valid_idx]
    
    return _format_examples(df_train.iloc[top_k_original_idx], name)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(name, examples):
    """Combine the base task prompt with formatted ICL examples."""
    system_prompt = PROMPTS[name]
    if not examples:
        return system_prompt

    from_col, to_col = TASK_COL_MAP[name]
    system_prompt += "\n\n" + "\n\n".join(
        f"{from_col}: {inp}\n{to_col}: {out}"
        for inp, out in examples
    )
    return system_prompt

def _create_prompt(sample_strategy, df_train, df_test, name, k):
    """Build parallel lists of (system_prompts, user_prompts)."""
    from_col, to_col = TASK_COL_MAP[name]
    test_inputs = df_test[from_col].tolist()

    # User prompt: present the input and ask for the output
    user_prompts = [f"{from_col}: {inp}\n{to_col}:" for inp in test_inputs]

    # Zero-shot — no examples
    if k == 0:
        system_prompt = _build_system_prompt(name, examples=None)
        return [system_prompt] * len(test_inputs), user_prompts

    # Random — same examples for every test molecule
    if sample_strategy == "random":
        system_prompt = _build_system_prompt(
            name, _random_sample_examples(df_train, k, name)
        )
        return [system_prompt] * len(test_inputs), user_prompts

    # Scaffold — examples depend on the test molecule
    if sample_strategy == "scaffold":
        if from_col != "smiles":
            raise ValueError(
                f"Scaffold sampling requires a SMILES input column, "
                f"but task '{name}' uses '{from_col}'. "
                f"Use 'random' sampling instead."
            )
            
        train_fps = [_scaffold_fingerprint(s) for s in df_train["smiles"]]
        system_prompts = [
            _build_system_prompt(
                name,
                _scaffold_sample_examples(df_train, train_fps, s, k, name),
            )
            for s in test_inputs
        ]
        return system_prompts, user_prompts

    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(name, sample_strategy="random", k=0):
    """Load the ChEBI-20 dataset and return prompts + test data."""
    if name not in TASK_COL_MAP:
        raise ValueError(
            f"Unknown task '{name}'. Choose from {list(TASK_COL_MAP.keys())}"
        )

    # 1) Load train, val, test directly into DataFrames
    dfs = {}
    for split, url in FILES.items():
        df = pd.read_csv(url, sep='\t')
        # Standardize column names to match TASK_COL_MAP
        df.rename(columns={"SMILES": "smiles", "CID": "cid"}, inplace=True)
        
        # Filter invalid SMILES and canonicalise
        dfs[split] = _canonicalise_smiles(df)

    df_train = dfs["train"]
    df_test = dfs["test"]

    if len(df_test) >= 100:
        df_test = df_test.sample(n=100, random_state=SEED).reset_index(drop=True)
    else:
        print(f"Warning: Test set has only {len(df_test)} valid samples. Using all available.")

    system_prompts, user_prompts = _create_prompt(
        sample_strategy=sample_strategy,
        df_train=df_train,
        df_test=df_test,
        name=name,
        k=k,
    )

    return system_prompts, user_prompts, df_test