"""
Task 2.2 analyzer for feature-selection comparison outputs.

This script reads candidate feature ranking tables generated as:
    f"{flow_type}" + "_{pred_obj}_candidate_features_top" + str(comp_top) + ".csv"

It computes three Task 2.2 metrics:
- ranking agreement (Kendall tau)
- symmetry bias (pair completion + side imbalance)
- locality bias (pedestrian continuity/locality)

Typical usage:
    python feature_selection_analyzer.py \
        --feat-path ../output_single_featimp/ \
        --flow-type entrance \
        --pred-objs step_x step_y \
        --comp-top 25 \
        --base-col top_shap \
        --comp-cols llm_selection_wo_val fused_importance \
        --top-k 25

Outputs:
    <out_prefix>_ranking_agreement.csv
    <out_prefix>_symmetry_bias.csv
    <out_prefix>_locality_bias.csv

Optional selective run:
    python feature_selection_analyzer.py ... --metric locality_bias

If `--metric` is omitted, all three outputs are generated.
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


@dataclass
class FeatureSelectionAnalyzer:
    """Compute Task 2.2 metrics from comp-top candidate feature tables."""

    feat_path: str
    flow_type: str
    pred_objs: List[str]
    comp_top: int
    base_col: str
    comp_cols: List[str]

    def __post_init__(self) -> None:
        if isinstance(self.pred_objs, str):
            self.pred_objs = [self.pred_objs]
        if isinstance(self.comp_cols, str):
            self.comp_cols = [self.comp_cols]

    # helper function: build comp-top file name from flow/target
    def _comp_top_name(self, pred_obj: str) -> str:
        # Uses the exact pattern requested by user.
        return f"{self.flow_type}" + f"_{pred_obj}_candidate_features_top" + str(self.comp_top) + ".csv"

    # helper function: load comp-top feature ranking table
    def _load_comp_table(self, pred_obj: str) -> pd.DataFrame:
        name = self._comp_top_name(pred_obj)
        path = os.path.join(self.feat_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature table not found: {path}")
        return pd.read_csv(path)

    # helper function: swap feature prefixes for symmetry checks
    def _swap_prefix(self, feat: str, swap_rules: List[Tuple[str, str]]) -> Optional[str]:
        if not isinstance(feat, str):
            return None
        for a, b in swap_rules:
            if feat.startswith(a):
                return b + feat[len(a):]
            if feat.startswith(b):
                return a + feat[len(b):]
        return None

    # helper function: extract zone prefix from feature name
    def _get_zone_prefix(self, feat: str, zones: List[str]) -> Optional[str]:
        if not isinstance(feat, str):
            return None
        for z in zones:
            if feat.startswith(f"{z}_"):
                return z
        return None

    # helper function: count features by zone
    def _zone_counts(self, ranked_feats: List[str], zones: List[str]) -> Dict[str, int]:
        counts = {z: 0 for z in zones}
        for feat in ranked_feats:
            zone = self._get_zone_prefix(feat, zones)
            if zone is not None:
                counts[zone] += 1
        return counts

    # helper function: compute normalized pair imbalance
    @staticmethod
    def _pair_imbalance(a: int, b: int) -> float:
        denom = a + b
        return (abs(a - b) / denom) if denom > 0 else np.nan

    # helper function: extract pedestrian index from feature name
    @staticmethod
    def _extract_ped_id(feat: str) -> Optional[int]:
        if not isinstance(feat, str):
            return None
        match = re.search(r"ped(\d+)", feat)
        return int(match.group(1)) if match else None

    # helper function: compute locality statistics from ranked features
    def _locality_stats(self, ranked_feats: List[str]) -> Dict[str, object]:
        ped_ids = [self._extract_ped_id(f) for f in ranked_feats]
        ped_ids = [pid for pid in ped_ids if pid is not None]
        unique_ids = sorted(set(ped_ids))

        if not unique_ids:
            return {
                "ped_ids_used": "",
                "ped_count_unique": 0,
                "ped_max": np.nan,
                "ped_prefix_depth": 0,
                "ped_gap_count": np.nan,
                "ped_continuity_score": np.nan,
            }

        ped_max = max(unique_ids)
        prefix_depth = 0
        for idx in range(1, ped_max + 1):
            if idx in unique_ids:
                prefix_depth += 1
            else:
                break

        gap_count = len([idx for idx in range(1, ped_max + 1) if idx not in unique_ids])
        continuity_score = (prefix_depth / ped_max) if ped_max > 0 else np.nan

        return {
            "ped_ids_used": ",".join(map(str, unique_ids)),
            "ped_count_unique": len(unique_ids),
            "ped_max": ped_max,
            "ped_prefix_depth": prefix_depth,
            "ped_gap_count": gap_count,
            "ped_continuity_score": continuity_score,
        }

    def ranking_agreement_table(self) -> pd.DataFrame:
        """
        Compute Kendall tau between base_col ranking and each comp_col ranking.
        """
        rows: List[Dict[str, object]] = []

        for pred_obj in self.pred_objs:
            df = self._load_comp_table(pred_obj)
            if self.base_col not in df.columns:
                raise ValueError(f"base_col '{self.base_col}' not found for target '{pred_obj}'.")

            base_rank = df[self.base_col].dropna().tolist()
            base_pos = {feat: i for i, feat in enumerate(base_rank)}

            for col in self.comp_cols:
                if col not in df.columns:
                    continue
                other_rank = df[col].dropna().tolist()
                aligned = [base_pos[f] for f in other_rank if f in base_pos]
                tau = kendalltau(range(len(aligned)), aligned).statistic if len(aligned) >= 2 else np.nan

                rows.append(
                    {
                        "target": pred_obj,
                        "base_method": self.base_col,
                        "comp_method": col,
                        "kendall_tau": float(tau) if tau is not None else np.nan,
                        "n_intersection": int(len(aligned)),
                    }
                )

        return pd.DataFrame(rows)

    def symmetry_bias_table(
        self,
        swap_rules: Optional[List[Tuple[str, str]]] = None,
        top_k: Optional[int] = None,
        rank_weighted: bool = True,
    ) -> pd.DataFrame:
        """
        Compute symmetry completion and side imbalance metrics.
        """
        if swap_rules is None:
            swap_rules = [("ll_", "rr_"), ("lf_", "rf_"), ("lb_", "rb_"), ("ff_", "bb_")]

        zones = ["ff", "bb", "rf", "lf", "rb", "lb", "rr", "ll"]
        zone_pairs = [("ff", "bb"), ("rf", "lf"), ("rb", "lb"), ("rr", "ll")]

        rows: List[Dict[str, object]] = []

        for pred_obj in self.pred_objs:
            df = self._load_comp_table(pred_obj)
            methods = [self.base_col] + list(self.comp_cols)
            methods = [m for m in methods if m in df.columns]

            for method in methods:
                ranked = df[method].dropna().tolist()
                if top_k is not None:
                    ranked = ranked[:top_k]

                counts = self._zone_counts(ranked, zones)

                pair_imbal: Dict[str, float] = {}
                pair_weights: Dict[str, int] = {}
                for a, b in zone_pairs:
                    ia = counts.get(a, 0)
                    ib = counts.get(b, 0)
                    pair_imbal[f"imb_{a}_{b}"] = self._pair_imbalance(ia, ib)
                    pair_weights[f"w_{a}_{b}"] = ia + ib

                total_w = sum(pair_weights.values())
                overall_imb = (
                    sum(pair_weights[k] * pair_imbal[k.replace("w_", "imb_")] for k in pair_weights.keys()) / total_w
                ) if total_w > 0 else np.nan

                feat_set = set(ranked)
                rank_pos = {f: i for i, f in enumerate(ranked)}

                seen_pairs = set()
                completed: List[Tuple[str, str]] = []
                singletons: List[Tuple[str, str]] = []

                for feat in ranked:
                    sym = self._swap_prefix(feat, swap_rules)
                    if sym is None:
                        continue

                    key = tuple(sorted([feat, sym]))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    if feat in feat_set and sym in feat_set:
                        completed.append((feat, sym))
                    else:
                        existing = feat if feat in feat_set else sym
                        missing = sym if feat in feat_set else feat
                        singletons.append((existing, missing))

                denom = len(completed) + len(singletons)
                sym_score = (len(completed) / denom) if denom > 0 else np.nan

                out: Dict[str, object] = {
                    "target": pred_obj,
                    "method": method,
                    "top_k": top_k if top_k is not None else len(ranked),
                    "pairs_completed": int(len(completed)),
                    "pairs_singletons": int(len(singletons)),
                    "symmetry_score": float(sym_score) if not np.isnan(sym_score) else np.nan,
                    "side_imbalance_overall": float(overall_imb) if not np.isnan(overall_imb) else np.nan,
                }

                out.update(counts)
                out.update(pair_imbal)

                if rank_weighted:
                    weights = []
                    for a, b in completed:
                        da = rank_pos.get(a, None)
                        db = rank_pos.get(b, None)
                        if da is None or db is None:
                            continue
                        weights.append(1.0 / (1.0 + abs(da - db)))
                    out["symmetry_score_weighted"] = (float(sum(weights)) / denom) if denom > 0 else np.nan

                rows.append(out)

        return (
            pd.DataFrame(rows)
            .sort_values(["target", "symmetry_score"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def locality_bias_table(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Compute pedestrian locality/continuity diagnostics per method.
        """
        rows: List[Dict[str, object]] = []

        for pred_obj in self.pred_objs:
            df = self._load_comp_table(pred_obj)
            methods = [self.base_col] + list(self.comp_cols)
            methods = [m for m in methods if m in df.columns]

            for method in methods:
                ranked = df[method].dropna().tolist()
                if top_k is not None:
                    ranked = ranked[:top_k]

                out: Dict[str, object] = {
                    "target": pred_obj,
                    "method": method,
                    "top_k": top_k if top_k is not None else len(ranked),
                }
                out.update(self._locality_stats(ranked))
                rows.append(out)

        return (
            pd.DataFrame(rows)
            .sort_values(["target", "ped_continuity_score", "ped_max"], ascending=[True, False, False])
            .reset_index(drop=True)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.2 analyzer: ranking agreement, symmetry bias, and locality bias")
    parser.add_argument("--feat-path", type=str, default="../output_single_featimp/", help="Path to candidate feature CSVs")
    parser.add_argument("--flow-type", type=str, required=True, help="Flow type, e.g. entrance")
    parser.add_argument("--pred-objs", nargs="+", default=["step_x", "step_y"], help="Prediction targets. Default: step_x step_y")
    parser.add_argument("--comp-top", type=int, default=25, help="Top-k used in comp table filename")
    parser.add_argument("--base-col", type=str, default="top_shap", help="Baseline ranking column")
    parser.add_argument(
        "--comp-cols",
        nargs="+",
        default=["llm_selection_wo_val", "fused_importance"],
        help="Comparison ranking columns. Default: llm_selection_wo_val fused_importance",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k slice for symmetry/locality")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["ranking_agreement", "symmetry_bias", "locality_bias"],
        help="Run/output only one metric. Default: run all three.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for result CSVs")
    parser.add_argument("--out-prefix", type=str, default=None, help="Prefix for output filenames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    analyzer = FeatureSelectionAnalyzer(
        feat_path=args.feat_path,
        flow_type=args.flow_type,
        pred_objs=args.pred_objs,
        comp_top=args.comp_top,
        base_col=args.base_col,
        comp_cols=args.comp_cols,
    )

    out_dir = args.out_dir if args.out_dir else args.feat_path
    os.makedirs(out_dir, exist_ok=True)

    prefix = args.out_prefix if args.out_prefix else f"{args.flow_type}"

    metrics_to_run = [args.metric] if args.metric else ["ranking_agreement", "symmetry_bias", "locality_bias"]
    saved_paths: List[str] = []

    if "ranking_agreement" in metrics_to_run:
        ranking_df = analyzer.ranking_agreement_table()
        ranking_path = os.path.join(out_dir, f"{prefix}_ranking_agreement.csv")
        ranking_df.to_csv(ranking_path, index=False)
        saved_paths.append(ranking_path)

    if "symmetry_bias" in metrics_to_run:
        symmetry_df = analyzer.symmetry_bias_table(top_k=args.top_k)
        symmetry_path = os.path.join(out_dir, f"{prefix}_symmetry_bias.csv")
        symmetry_df.to_csv(symmetry_path, index=False)
        saved_paths.append(symmetry_path)

    if "locality_bias" in metrics_to_run:
        locality_df = analyzer.locality_bias_table(top_k=args.top_k)
        locality_path = os.path.join(out_dir, f"{prefix}_locality_bias.csv")
        locality_df.to_csv(locality_path, index=False)
        saved_paths.append(locality_path)

    print("Saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
