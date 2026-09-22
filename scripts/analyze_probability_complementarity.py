from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHOICES = ["a", "b", "c", "d"]
PROB_COLS = [f"p_{c}" for c in CHOICES]


def validate_probs(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"id", *PROB_COLS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")

    out = frame[["id", *PROB_COLS]].copy()
    if out["id"].duplicated().any():
        dupes = int(out["id"].duplicated().sum())
        raise ValueError(f"{name} has {dupes} duplicate ids")

    values = out[PROB_COLS].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite probability values")
    if (values.to_numpy() < 0).any():
        raise ValueError(f"{name} contains negative probability values")

    sums = values.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError(f"{name} contains rows with non-positive probability sum")

    out[PROB_COLS] = values.div(sums, axis=0)
    return out


def entropy4(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only complementarity analysis between two four-choice probability files."
    )
    parser.add_argument("--reference-probs", type=Path, required=True)
    parser.add_argument("--candidate-probs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.80,
        help="Threshold for max normalized choice score.",
    )
    args = parser.parse_args()

    ref = validate_probs(
        pd.read_csv(args.reference_probs, encoding="utf-8-sig"),
        args.reference_name,
    )
    cand = validate_probs(
        pd.read_csv(args.candidate_probs, encoding="utf-8-sig"),
        args.candidate_name,
    )

    ref_ids = set(ref["id"].astype(str))
    cand_ids = set(cand["id"].astype(str))
    if ref_ids != cand_ids:
        raise ValueError(
            "ID sets differ: "
            f"reference_only={len(ref_ids-cand_ids)}, "
            f"candidate_only={len(cand_ids-ref_ids)}"
        )

    merged = ref.merge(cand, on="id", suffixes=("_ref", "_cand"), validate="one_to_one")

    p_ref = merged[[f"{c}_ref" for c in PROB_COLS]].to_numpy(dtype=float)
    p_cand = merged[[f"{c}_cand" for c in PROB_COLS]].to_numpy(dtype=float)

    ref_idx = p_ref.argmax(axis=1)
    cand_idx = p_cand.argmax(axis=1)
    ref_pred = np.array(CHOICES, dtype=object)[ref_idx]
    cand_pred = np.array(CHOICES, dtype=object)[cand_idx]
    disagree = ref_idx != cand_idx

    ref_conf = p_ref.max(axis=1)
    cand_conf = p_cand.max(axis=1)
    both_high = (ref_conf >= args.high_confidence) & (cand_conf >= args.high_confidence)
    high_disagree = disagree & both_high

    ent_ref = entropy4(p_ref)
    ent_cand = entropy4(p_cand)
    js = js_divergence(p_ref, p_cand)

    out = pd.DataFrame({
        "id": merged["id"],
        "reference_prediction": ref_pred,
        "candidate_prediction": cand_pred,
        "disagree": disagree,
        "reference_confidence": ref_conf,
        "candidate_confidence": cand_conf,
        "reference_entropy": ent_ref,
        "candidate_entropy": ent_cand,
        "js_divergence": js,
        "both_high_confidence": both_high,
        "high_confidence_disagreement": high_disagree,
    })

    for i, choice in enumerate(CHOICES):
        out[f"ref_p_{choice}"] = p_ref[:, i]
        out[f"cand_p_{choice}"] = p_cand[:, i]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        args.output_dir / "paired_probability_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    out[out["disagree"]].sort_values(
        ["high_confidence_disagreement", "js_divergence"],
        ascending=[False, False],
    ).to_csv(
        args.output_dir / "disagreements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr = np.corrcoef(p_ref.reshape(-1), p_cand.reshape(-1))[0, 1]
    summary = {
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "rows": int(len(out)),
        "agreement_count": int((~disagree).sum()),
        "disagreement_count": int(disagree.sum()),
        "disagreement_rate": float(disagree.mean()),
        "high_confidence_threshold": float(args.high_confidence),
        "both_high_confidence_count": int(both_high.sum()),
        "high_confidence_disagreement_count": int(high_disagree.sum()),
        "high_confidence_disagreement_rate_all_rows": float(high_disagree.mean()),
        "mean_reference_confidence": float(ref_conf.mean()),
        "mean_candidate_confidence": float(cand_conf.mean()),
        "mean_reference_entropy": float(ent_ref.mean()),
        "mean_candidate_entropy": float(ent_cand.mean()),
        "mean_js_divergence": float(js.mean()),
        "median_js_divergence": float(np.median(js)),
        "flattened_probability_pearson_correlation": float(corr),
        "note": (
            "This is a test-set diversity/complementarity diagnostic only. "
            "Without test gold, disagreements cannot be labeled as rescues or regressions."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
