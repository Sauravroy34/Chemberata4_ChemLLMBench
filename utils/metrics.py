import ast
import evaluate
import numpy as np
import Levenshtein
from fcd import get_fcd, load_ref_model, canonical_smiles
from nltk.translate.meteor_score import meteor_score
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast


TOKENIZER = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
FCD_MODEL = load_ref_model()
BLEU = evaluate.load("bleu")
ROUGE = evaluate.load("rouge")
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def wp(s):
    return TOKENIZER.tokenize(s, truncation=True, max_length=512)


def _is_invalid_prediction(smiles):
    return not smiles or Chem.MolFromSmiles(smiles) is None


def _normalise(smiles):
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else ""


def _canon(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except Exception:
        return None


def _is_invalid_retro_prediction(smiles):
    if not smiles:
        return True
    for frag in smiles.split("."):
        if not frag or Chem.MolFromSmiles(frag) is None:
            return True
    return False


def _normalise_retro(smiles):
    if not smiles:
        return ""
    canon = []
    for frag in smiles.split("."):
        mol = Chem.MolFromSmiles(frag) if frag else None
        if mol is None:
            return ""
        canon.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon))


def _compute_fcd(pre_smiles, true_smiles):

    canon_pred_smiles = [w for w in canonical_smiles(pre_smiles) if w is not None]
    canon_true_smiles = [w for w in canonical_smiles(true_smiles) if w is not None]

    if len(canon_pred_smiles) < 2 or len(canon_true_smiles) < 2:
        return float("nan")

    return float(get_fcd(canon_pred_smiles, canon_true_smiles, model=FCD_MODEL))


def _fingerprint_similarities(mol_a, mol_b):
    maccs = DataStructs.TanimotoSimilarity(
        MACCSkeys.GenMACCSKeys(mol_a), MACCSkeys.GenMACCSKeys(mol_b))
    rdk = DataStructs.TanimotoSimilarity(
        Chem.RDKFingerprint(mol_a), Chem.RDKFingerprint(mol_b))
    morgan = DataStructs.TanimotoSimilarity(
        MORGAN_GEN.GetSparseCountFingerprint(mol_a),
        MORGAN_GEN.GetSparseCountFingerprint(mol_b))
    return maccs, rdk, morgan


def compute_caption_metrices(y_true, y_pred):
    refs = [[t] for t in y_true]
    bleu2 = BLEU.compute(predictions=y_pred, references=refs, max_order=2, tokenizer=wp)["bleu"]
    bleu4 = BLEU.compute(predictions=y_pred, references=refs, max_order=4, tokenizer=wp)["bleu"]
    r = ROUGE.compute(predictions=y_pred, references=y_true, use_stemmer=False)
    m = np.mean([meteor_score([wp(t)], wp(p)) for t, p in zip(y_true, y_pred)])

    return {
        "bleu2": float(bleu2),
        "bleu4": float(bleu4),
        "rouge1": float(r["rouge1"]),
        "rouge2": float(r["rouge2"]),
        "rougeL": float(r["rougeL"]),
        "meteor": float(m),
        "total": len(y_true),
    }


def compute_design_metrices(y_true, y_pred):
    refs = [[t] for t in y_true]
    lev = float(np.mean([Levenshtein.distance(t, p) for t, p in zip(y_true, y_pred)]))
    bleu_design = BLEU.compute(predictions=y_pred, references=refs, tokenizer=wp)["bleu"]

    excat_match = 0
    maccs_s, rdk_s, morgan_s = [], [], []
    valid_pred_can, true_can = [], []

    for t, p in zip(y_true, y_pred):
        mol_t = Chem.MolFromSmiles(t) if t else None
        mol_p = Chem.MolFromSmiles(p) if p else None

        if mol_t is not None:
            true_can.append(Chem.MolToSmiles(mol_t))

        if mol_p is None:
            continue

        valid_pred_can.append(Chem.MolToSmiles(mol_p))

        if mol_t is None:
            # reference itself is unparsable: nothing to compare against
            continue

        try:
            if Chem.MolToInchi(mol_t) == Chem.MolToInchi(mol_p):
                excat_match += 1
        except Exception:
            pass

        m, r, mo = _fingerprint_similarities(mol_t, mol_p)
        maccs_s.append(m)
        rdk_s.append(r)
        morgan_s.append(mo)

    mean_or_zero = lambda xs: float(np.mean(xs)) if xs else 0.0

    return {
        "bleu": float(bleu_design),
        "exact": excat_match / len(y_true),
        "levenshtein": lev,
        "validity": len(valid_pred_can) / len(y_true),
        "maccs_fts": mean_or_zero(maccs_s),  
        "rdk_fts": mean_or_zero(rdk_s),
        "morgan_fts": mean_or_zero(morgan_s),
        "fcd": _compute_fcd(valid_pred_can, true_can),
        "valid_predictions": len(valid_pred_can),
        "total": len(y_true),
    }


def compute_name_predictions(y_true, y_pred, task_name):
    if task_name in ("iupac2smiles",):
        correct = 0
        for gt, pred in zip(y_true, y_pred):
            gt_can = _canon(gt) or gt
            pred_can = _canon(pred)
            if pred_can is not None and pred_can == gt_can:
                correct += 1
    else:
        correct = sum(gt == pred for gt, pred in zip(y_true, y_pred))

    return {
        "task": task_name,
        "accuracy": correct / len(y_true),
        "correct": correct,
        "total": len(y_true),
    }


def _flip_invalid(y_true, y_pred):
    return [1 - yt if yp == -1 else yp for yt, yp in zip(y_true, y_pred)]


def compute_property_predictions(y_true, y_pred):
    n_invalid = sum(yp == -1 for yp in y_pred)
    y_pred_clean = _flip_invalid(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred_clean),
        "f1": f1_score(y_true, y_pred_clean, average="macro", zero_division=np.nan),
        "total": len(y_true),
        "valid_predictions": len(y_true) - n_invalid,
        "invalid_predictions": n_invalid,
    }


def compute_yeild_predictions(y_true, y_pred):
    n_invalid = sum(yp == -1 for yp in y_pred)
    y_pred_clean = _flip_invalid(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred_clean),
        "total": len(y_true),
        "valid_predictions": len(y_true) - n_invalid,
        "invalid_predictions": n_invalid,
    }


def compute_reaction_predictions(y_true, y_pred):
    n = len(y_true)
    num_invalid = sum(_is_invalid_prediction(p) for p in y_pred)

    norm_true = [_normalise(t) for t in y_true]
    norm_pred = [_normalise(p) for p in y_pred]
    correct_flags = [bool(p) and t in p.split(".") for t, p in zip(norm_true, norm_pred)]
    num_correct = sum(correct_flags)

    metrics = {
        "top1_accuracy": num_correct / n if n else 0.0,
        "invalid_smiles_percent": 100.0 * num_invalid / n if n else 0.0,
        "correct": num_correct,
        "invalid": num_invalid,
        "total": n,
    }
    return metrics, correct_flags


def compute_reaction_selection(y_true, y_pred, candidate_rank, task):
    n = len(y_true)
    norm_pred = [_normalise(p) for p in y_pred]

    if task.lower() != "ligand_selection":
        norm_true = [_normalise(t) for t in y_true]
        correct_flags = [bool(p) and p == t for t, p in zip(norm_true, norm_pred)]
    else:
        # top-50% for ligand selection
        ranks = [ast.literal_eval(r) for r in candidate_rank]
        top_half = [[_normalise(s) for s in r[:len(r) // 2]] for r in ranks]
        correct_flags = [bool(p) and p in t for t, p in zip(top_half, norm_pred)]

    num_correct = sum(correct_flags)
    metrics = {
        "accuracy": num_correct / n if n else 0.0,
        "correct": num_correct,
        "total": n,
    }
    return metrics, correct_flags

def compute_retrosynthesis(y_true, y_pred):
    n = len(y_true)
    num_invalid = sum(_is_invalid_retro_prediction(p) for p in y_pred)

    norm_true = [_normalise_retro(t) for t in y_true]
    norm_pred = [_normalise_retro(p) for p in y_pred]
    correct_flags = [bool(p) and p == t for t, p in zip(norm_true, norm_pred)]
    num_correct = sum(correct_flags)

    metrics = {
        "top1_accuracy": num_correct / n if n else 0.0,
        "invalid_smiles_percent": 100.0 * num_invalid / n if n else 0.0,
        "correct": num_correct,
        "invalid": num_invalid,
        "total": n,
    }
    return metrics, correct_flags