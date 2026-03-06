#%%% import packages 
import os
import re
from fastavro import writer
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import lightgbm as lgb
from lightgbm.basic import LightGBMError

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import lru_cache

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kendalltau

import warnings
warnings.filterwarnings("ignore")

#%%% define data class for use
@dataclass
class SelectionMechanismAnalyzer:
    data_path: str
    model_path: str
    feat_path: str
    flow_type: str
    pred_objs: List[str]
    base_top: int
    comp_top: int
    base_col: str
    comp_cols: List[str]

    n_splits: int = 5
    random_state: int = 42
    bootstrap_B: int = 10
    empirical_importance_type: str = "gain"
    include_shap_if_available: bool = False
    n_jobs: int = 1                  # workers for parallel loops
    show_progress: bool = False      # enable/disable tqdm
    cache_data: bool = True          # cache I/O results

    noise_sigmas: Optional[List[float]] = None          # e.g. [0.0, 0.1, 0.2]
    train_fracs: Optional[List[float]] = None           # e.g. [1.0, 0.7, 0.4]

    def __post_init__(self) -> None:
        self.file_name = f"ds_vector_{self.flow_type}.csv"
        self.base_top_name_tpl = f"{self.flow_type}" + "_{pred_obj}_candidate_features_top" + str(self.base_top) + ".csv"
        self.comp_top_name_tpl = f"{self.flow_type}" + "_{pred_obj}_candidate_features_top" + str(self.comp_top) + ".csv"

        if isinstance(self.comp_cols, str):
            self.comp_cols = [self.comp_cols]
        if isinstance(self.pred_objs, str):
            self.pred_objs = [self.pred_objs]
        if self.noise_sigmas is None:
            self.noise_sigmas = [0.0, 0.1, 0.2]
        if self.train_fracs is None:
            self.train_fracs = [1.0, 0.7, 0.4]
        
        # Color palette for consistent visualization
        self.method_colors = {
            "top_shap": "#FF8C00",              # orange
            "llm_selection": "#2CA02C",        # green
            "fused_importance": "#1F77B4",     # blue
        }

        # prepare caches if requested
        self._df_cache: Optional[pd.DataFrame] = None
        self._params_cache: Dict[str, dict] = {}
        self._feat_table_cache: Dict[str, pd.DataFrame] = {}

    # helper function: optional progress iterator
    def _iter(
        self,
        iterable: Iterable[Any],
        desc: Optional[str] = None,
        leave: bool = True,
        total: Optional[int] = None,
        position: Optional[int] = None,
        **tqdm_kwargs: Any,
    ) -> Iterable[Any]:
        """
        Helper for progress bars with custom description.
        Usage: for x in self._iter(..., desc="Target: ...", ...):
        """
        if self.show_progress:
            return tqdm(iterable, desc=desc, leave=leave, total=total, position=position, **tqdm_kwargs)
        return iterable
    
    # helper function: normalize display name for methods
    def _normalize_method_name(self, method: str) -> str:
        """Normalize method names for consistent display.
        Converts any llm_selection_* variant to just 'llm_selection'.
        """
        if "llm_selection" in method:
            return "llm_selection"
        return method
    
    # helper function: get plotting color for a method
    def _get_method_color(self, method: str) -> str:
        """Get color for a given method, using normalized name."""
        normalized = self._normalize_method_name(method)
        return self.method_colors.get(normalized, "#000000")  # default to black if not found
    
    # helper function: load and cache full dataset
    def _load_data_full(self) -> pd.DataFrame:
        # cache the loaded DataFrame to avoid repeated disk I/O
        if self.cache_data and self._df_cache is not None:
            return self._df_cache

        df = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df = df.drop(columns=["id", "time"], errors="ignore")
        if self.cache_data:
            self._df_cache = df
        return df

    # helper function: split dataframe into features and target
    def _get_Xy(self, df: pd.DataFrame, pred_obj: str) -> Tuple[pd.DataFrame, pd.Series]:
        y = df[pred_obj]
        X = df.drop(columns=["step_x", "step_y"], errors="ignore")
        return X, y

    # helper function: load and cache model params
    def _load_params(self, pred_obj: str) -> dict:
        # cache parameter files per target
        if self.cache_data and pred_obj in self._params_cache:
            return self._params_cache[pred_obj]

        param_name = f"{self.flow_type}_{pred_obj}_lgb_param.joblib"
        params = joblib.load(os.path.join(self.model_path, param_name))
        if not isinstance(params, dict):
            raise ValueError("Loaded parameters are not in dictionary format.")
        if self.cache_data:
            self._params_cache[pred_obj] = params
        return params

    # helper function: train/evaluate LightGBM on selected features
    def _train_eval_lgb(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, params: dict, feature_list: List[str], early_stopping_rounds: int = 100, verbose_eval: int = 0,) -> Dict[str, float]:
        p = params.copy()
        p["verbosity"] = -1
        p["device"] = "gpu"

        def _sanitize_params(p_local: dict) -> dict:
            p_local = p_local.copy()
            if str(p_local.get("device", "")).lower() == "gpu":
                max_bin = int(p_local.get("max_bin", 255))
                if max_bin > 200:
                    p_local["max_bin"] = 200
                p_local["gpu_use_dp"] = False
            return p_local

        def _build_dataset_params(p_local: dict) -> dict:
            ds_params = {"max_bin": int(p_local.get("max_bin", 200))}
            if "min_data_in_bin" in p_local:
                ds_params["min_data_in_bin"] = p_local["min_data_in_bin"]
            return ds_params

        def _train_once(p_local: dict) -> lgb.Booster:
            ds_params = _build_dataset_params(p_local)
            dtrain = lgb.Dataset(
                X_train[feature_list],
                label=y_train,
                params=ds_params,
                free_raw_data=False,
            )
            dvalid = lgb.Dataset(
                X_test[feature_list],
                label=y_test,
                reference=dtrain,
                params=ds_params,
                free_raw_data=False,
            )
            return lgb.train(
                params=p_local,
                train_set=dtrain,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=bool(verbose_eval)),
                    lgb.log_evaluation(period=verbose_eval) if verbose_eval else lgb.log_evaluation(period=0),
                ],
            )

        p = _sanitize_params(p)
        try:
            model = _train_once(p)
        except LightGBMError as e:
            msg = str(e).lower()
            if "bin size" in msg and "gpu" in msg:
                p_cpu = p.copy()
                p_cpu["device"] = "cpu"
                p_cpu.pop("gpu_use_dp", None)
                model = _train_once(p_cpu)
            else:
                raise

        preds = model.predict(X_test[feature_list], num_iteration=model.best_iteration)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        return {"rmse": rmse, "r2": r2, "best_iteration": int(model.best_iteration)}

    # helper function: derive baseline and add-on feature lists
    def _ordered_features_from_table(self, pred_obj: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        baseline = top{base_top} from base_col (common features)
        For each method col, build add_list = features in top{comp_top} of that col but not in baseline.
        """
        base_top_name = self.base_top_name_tpl.format(pred_obj=pred_obj)
        comp_top_name = self.comp_top_name_tpl.format(pred_obj=pred_obj)
        # cache feature tables to avoid repeated reads
        if self.cache_data and base_top_name in self._feat_table_cache:
            df_base = self._feat_table_cache[base_top_name]
        else:
            df_base = pd.read_csv(os.path.join(self.feat_path, base_top_name))
            if self.cache_data:
                self._feat_table_cache[base_top_name] = df_base

        if self.cache_data and comp_top_name in self._feat_table_cache:
            df_comp = self._feat_table_cache[comp_top_name]
        else:
            df_comp = pd.read_csv(os.path.join(self.feat_path, comp_top_name))
            if self.cache_data:
                self._feat_table_cache[comp_top_name] = df_comp

        baseline = df_base[self.base_col].dropna().tolist()

        add_dict = {}
        for col in self.comp_cols:
            if col not in df_comp.columns:
                raise ValueError(f"Column '{col}' not found in {self.comp_top_name}")
            feat_list = df_comp[col].dropna().tolist()
            add_dict[col] = [f for f in feat_list if f not in baseline]

        # also include base_col extension itself (top{comp_top} base_col minus baseline)
        if self.base_col in df_comp.columns:
            base_extended = df_comp[self.base_col].dropna().tolist()
            add_dict[self.base_col] = [f for f in base_extended if f not in baseline]

        return baseline, add_dict

    # helper function: generate incremental feature sets
    def _incremental_feature_sets(
        self,
        base: List[str],
        to_add_in_order: List[str],
        max_add: Optional[int],
    ) -> Iterator[Tuple[int, List[str]]]:
        if max_add is None:
            max_add = len(to_add_in_order)
        for k in range(0, max_add + 1):
            yield k, base + to_add_in_order[:k]

    # helper function: process one CV fold for incremental evaluation
    def _process_fold(self, X: pd.DataFrame, y: pd.Series, params: dict,
                      baseline: List[str], add_list: List[str], method: str,
                      fold: int, tr: np.ndarray, te: np.ndarray,
                      max_add_use: int) -> List[Dict[str, Any]]:
        """Compute all k-added evaluations for a single fold (returns rows list)."""
        rows = []
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        for k_added, feat_list in self._incremental_feature_sets(baseline, add_list, max_add=max_add_use):
            m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te, params, feat_list)
            rows.append({
                "target": None,  # filled by caller
                "method": method,
                "fold": fold,
                "k_added": k_added,
                "n_features": len(feat_list),
                "rmse": m["rmse"],
                "r2": m["r2"],
            })
        return rows

    # helper function: add Gaussian noise to target
    def _apply_target_noise(self, y: pd.Series, sigma: float, seed: int) -> pd.Series:
        if sigma == 0.0:
            return y
        rng = np.random.RandomState(seed)
        return y + rng.normal(loc=0.0, scale=sigma, size=len(y))

    # helper function: swap feature prefixes for symmetry checks
    def _swap_prefix(self, feat: str, swap_rules: List[Tuple[str, str]]) -> Optional[str]:
        """
        Swap feature prefix according to rules. Returns swapped feature name or None if no rule matched.
        Example: ll_xxx <-> rr_xxx ; lf_xxx <-> rf_xxx
        """
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
        """
        Extract zone prefix from feature name, assuming pattern like 'll_xxx', 'rf_xxx', etc.
        Returns zone string if matched, else None.
        """
        if not isinstance(feat, str):
            return None
        for z in zones:
            if feat.startswith(f"{z}_"):
                return z
        return None

    # helper function: count features by zone
    def _zone_counts(self, ranked_feats: List[str], zones: List[str]) -> Dict[str, int]:
        counts = {z: 0 for z in zones}
        for f in ranked_feats:
            z = self._get_zone_prefix(f, zones)
            if z is not None:
                counts[z] += 1
        return counts

    # helper function: compute normalized pair imbalance
    def _pair_imbalance(self, a: int, b: int) -> float:
        """
        Normalized imbalance within a pair. Range [0, 1].
        0 means perfectly balanced; 1 means all mass on one side.
        """
        denom = a + b
        return (abs(a - b) / denom) if denom > 0 else np.nan

    # helper function: extract pedestrian index from feature name
    def _extract_ped_id(self, feat: str) -> Optional[int]:
        """
        Extract pedestrian index from feature name, e.g. ff_ped3_y -> 3.
        Returns None if no ped index is present.
        """
        if not isinstance(feat, str):
            return None
        m = re.search(r"ped(\d+)", feat)
        return int(m.group(1)) if m else None

    # helper function: compute pedestrian continuity statistics
    def _ped_stats_from_ranked(self, ranked_feats: List[str]) -> Dict[str, Any]:
        """
        Compute ped continuity/locality diagnostics from a ranked feature list.
        """
        ped_ids = []
        for f in ranked_feats:
            pid = self._extract_ped_id(f)
            if pid is not None:
                ped_ids.append(pid)

        ped_ids_unique = sorted(set(ped_ids))

        if len(ped_ids_unique) == 0:
            return {
                "ped_ids_used": "",
                "ped_count_unique": 0,
                "ped_max": np.nan,
                "ped_prefix_depth": 0,
                "ped_gap_count": np.nan,
                "ped_continuity_score": np.nan,
            }

        ped_max = max(ped_ids_unique)

        # largest consecutive prefix starting from 1
        prefix_depth = 0
        for i in range(1, ped_max + 1):
            if i in ped_ids_unique:
                prefix_depth += 1
            else:
                break

        gap_count = len([i for i in range(1, ped_max + 1) if i not in ped_ids_unique])

        continuity_score = (prefix_depth / ped_max) if ped_max > 0 else np.nan

        return {
            "ped_ids_used": ",".join(map(str, ped_ids_unique)),
            "ped_count_unique": len(ped_ids_unique),
            "ped_max": ped_max,
            "ped_prefix_depth": prefix_depth,
            "ped_gap_count": gap_count,
            "ped_continuity_score": continuity_score,
        }

    # helper function: fit full booster for empirical ranking
    def _fit_booster_full(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> lgb.Booster:
        p = params.copy()
        p["verbosity"] = -1
        if "device" not in p:
            p["device"] = "gpu"

        if str(p.get("device", "")).lower() == "gpu":
            p["max_bin"] = min(int(p.get("max_bin", 255)), 200)
            p["gpu_use_dp"] = False

        def _train_once(p_local: dict) -> lgb.Booster:
            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                params={"max_bin": int(p_local.get("max_bin", 200))},
                free_raw_data=False,
            )
            return lgb.train(params=p_local, train_set=dtrain)

        try:
            return _train_once(p)
        except LightGBMError as e:
            msg = str(e).lower()
            if "bin size" in msg and "gpu" in msg:
                p_cpu = p.copy()
                p_cpu["device"] = "cpu"
                p_cpu.pop("gpu_use_dp", None)
                return _train_once(p_cpu)
            raise

    # -------------------- helper functions: plotting --------------------
    # helper function: plot CV mean metric curves with std bands
    def plot_mean_curve_with_band(self, curve_df: pd.DataFrame, metric: str = "rmse", save: bool = True) -> None:
        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")
        if "target" in curve_df.columns:
            targets = sorted(curve_df["target"].dropna().unique())
        else:
            targets = [self.pred_objs[0]]

        for target in targets:
            d0 = curve_df[curve_df["target"] == target].copy() if "target" in curve_df.columns else curve_df.copy()
            if d0.empty:
                continue

            agg = (
                d0.groupby(["method", "n_features"])
                .agg(mean=(metric, "mean"), std=(metric, "std"))
                .reset_index()
                .sort_values(["method", "n_features"])
            )

            plt.figure(figsize=(10, 6))
            for method in sorted(agg["method"].unique()):
                d = agg[agg["method"] == method].sort_values("n_features")
                x = d["n_features"].values
                y = d["mean"].values
                s = np.nan_to_num(d["std"].values, nan=0.0)

                # Use normalized name for label and method color
                label = self._normalize_method_name(method)
                color = self._get_method_color(method)

                plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
                plt.fill_between(x, y - s, y + s, alpha=0.20, color=color)

            plt.title(f"{self.flow_type} | {target} | CV Mean {metric} vs n_features")
            plt.xlabel("n_features")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend()

            if save:
                name = f"{self.flow_type}_{target}_CVMean_{metric}.png"
                path = os.path.join(self.feat_path, name)
                plt.tight_layout()
                plt.savefig(path, dpi=400)

            # force x-axis major ticks every 1
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

            plt.show()

    # helper function: plot delta-k metric curves with std bands
    def plot_delta_k_with_band(self, delta_summary_df: pd.DataFrame, metric: str = "rmse", save: bool = True) -> None:
        if "target" in delta_summary_df.columns:
            targets = sorted(delta_summary_df["target"].dropna().unique())
        else:
            targets = [self.pred_objs[0]]

        for target in targets:
            d0 = delta_summary_df[delta_summary_df["target"] == target].copy() if "target" in delta_summary_df.columns else delta_summary_df.copy()
            if d0.empty:
                continue

            plt.figure(figsize=(10, 6))

            for method in sorted(d0["method"].unique()):
                d = d0[d0["method"] == method].sort_values("n_features")
                x = d["n_features"].values
                y = d["delta_mean"].values
                s = np.nan_to_num(d["delta_std"].values, nan=0.0)

                # Use normalized name for label and method color
                label = self._normalize_method_name(method)
                color = self._get_method_color(method)

                plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
                plt.fill_between(x, y - s, y + s, alpha=0.20, color=color)

            plt.axhline(0, linewidth=1, alpha=0.6)
            plt.title(f"{self.flow_type} | {target} | Δ{metric} (CV mean±std) vs n_features")
            plt.xlabel("n_features (k)")
            plt.ylabel(f"Δ{metric} = {metric}(k) - {metric}(k-1)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            if save:
                name = f"{self.flow_type}_{target}_CVDelta_{metric}.png"
                path = os.path.join(self.feat_path, name)
                plt.tight_layout()
                plt.savefig(path, dpi=400)

            # force x-axis major ticks every 1
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

            plt.show()

    # helper function: plot sensitivity curves by perturbation level
    def plot_sensitivity_mean_curves(self, sens_summary_df: pd.DataFrame, metric: str = "rmse", save: bool = True, max_panels: int = 6) -> None:
        """
        Plot mean curves for each perturb_value (up to max_panels), lines per method.
        """
        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        perturb_type = sens_summary_df["perturb_type"].iloc[0]
        vals = sorted(sens_summary_df["perturb_value"].unique())[:max_panels]

        for target in sorted(sens_summary_df["target"].unique()):
            for pv in vals:
                d0 = sens_summary_df[(sens_summary_df["target"] == target) & (sens_summary_df["perturb_value"] == pv)].copy()
                if d0.empty:
                    continue
                plt.figure(figsize=(10, 6))

                for method in sorted(d0["method"].unique()):
                    d = d0[d0["method"] == method].sort_values("n_features")
                    x = d["n_features"].values
                    y = d[mean_col].values
                    s = np.nan_to_num(d[std_col].values, nan=0.0)

                    # Use normalized name for label and method color
                    label = self._normalize_method_name(method)
                    color = self._get_method_color(method)

                    plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
                    plt.fill_between(x, y - s, y + s, alpha=0.20, color=color)

                plt.title(f"{self.flow_type} | {target} | {metric} vs n_features | {perturb_type}={pv}")
                plt.xlabel("n_features")
                plt.ylabel(metric)
                plt.grid(True, alpha=0.3)
                plt.legend()

                if save:
                    name = f"{self.flow_type}_{target}_{metric}_{perturb_type}_{pv}.png"
                    path = os.path.join(self.feat_path, name)
                    plt.tight_layout()
                    plt.savefig(path, dpi=400)

                plt.show()

    # helper function: provide Task 2 method navigation index
    @staticmethod
    def task_2_function_index() -> Dict[str, List[str]]:
        """Navigation helper for Task 2.1-2.5 public methods."""
        return {
            "2.1": ["effective_snr_by_method_cv"],
            "2.2": ["ranking_agreement_table", "symmetry_bias_table", "ped_continuity_table"],
            "2.3": ["bootstrap_empirical_ranking_stability"],
            "2.4": ["delta_k_from_curve_cv"],
            "2.5": ["sensitivity_curves_cv", "summarize_sensitivity_curves"],
        }

    # ==================== 2.1 Effective SNR by method ====================
    def effective_snr_by_method_cv(self, curve_df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses CV mean R2 at each n_features:
            SNR = R2 / (1-R2)
        Returns method-level effective SNR summary at the largest n_features available per method.
        """
        rows = []
        d = curve_df.copy()

        # choose terminal point per method and target as "effective snr"
        for target in sorted(d["target"].unique()):
            for method in sorted(d[d["target"] == target]["method"].unique()):
                dm = d[(d["target"] == target) & (d["method"] == method)]
                # aggregate mean curve
                agg = dm.groupby("n_features").agg(r2_mean=("r2", "mean"), r2_std=("r2", "std")).reset_index()
                agg = agg.sort_values("n_features")
                last = agg.iloc[-1]
                r2m = float(last["r2_mean"])
                snr = float(r2m / (1 - r2m)) if r2m < 1 else np.inf
                rows.append({
                    "target": target,
                    "method": method,
                    "terminal_n_features": int(last["n_features"]),
                    "cv_r2_mean": r2m,
                    "cv_r2_std": float(last["r2_std"]) if not np.isnan(last["r2_std"]) else 0.0,
                    "effective_snr_r2_over_1_minus_r2": snr,
                })

        return pd.DataFrame(rows)

    # ==================== 2.2 Ranking agreement/symmetry bias and imbalance/pedestrian-order continuity and locality bias (SHAP vs LLM vs fused) ====================
    def ped_continuity_table(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Measures whether a method selects pedestrian-indexed features continuously
        from ped1 outward (local-neighbor continuity).
        
        Output per target x method:
            target | method | top_k | ped_ids_used | ped_count_unique | ped_max
            ped_prefix_depth | ped_gap_count | ped_continuity_score
        """
        rows = []

        for pred_obj in self.pred_objs:
            comp_top_name = self.comp_top_name_tpl.format(pred_obj=pred_obj)
            df = pd.read_csv(os.path.join(self.feat_path, comp_top_name))

            methods_to_check = [self.base_col] + list(self.comp_cols)
            methods_to_check = [m for m in methods_to_check if m in df.columns]

            for method in methods_to_check:
                ranked = df[method].dropna().tolist()
                if top_k is not None:
                    ranked = ranked[:top_k]

                ped_stats = self._ped_stats_from_ranked(ranked)

                out = {
                    "target": pred_obj,
                    "method": method,
                    "top_k": top_k if top_k is not None else len(ranked),
                }
                out.update(ped_stats)
                rows.append(out)

        return (
            pd.DataFrame(rows)
            .sort_values(["target", "ped_continuity_score", "ped_max"], ascending=[True, False, False])
            .reset_index(drop=True)
        )

    def symmetry_bias_table(self, swap_rules: Optional[List[Tuple[str, str]]] = None, top_k: Optional[int] = None, rank_weighted: bool = True,) -> pd.DataFrame:
        """
        Measures whether a method tends to select symmetric semantic pairs together + side imbalance metrics.
        """
        if swap_rules is None:
            swap_rules = [("ll_", "rr_"), ("lf_", "rf_"), ("lb_", "rb_"), ("ff_", "bb_")]

        zones = ["ff", "bb", "rf", "lf", "rb", "lb", "rr", "ll"]
        zone_pairs = [("ff", "bb"), ("rf", "lf"), ("rb", "lb"), ("rr", "ll")]

        rows = []
        for pred_obj in self.pred_objs:
            comp_top_name = self.comp_top_name_tpl.format(pred_obj=pred_obj)
            df = pd.read_csv(os.path.join(self.feat_path, comp_top_name))

            # evaluate base_col + comp_cols (same scope as 2.2)
            methods_to_check = [self.base_col] + list(self.comp_cols)
            methods_to_check = [m for m in methods_to_check if m in df.columns]

            for method in methods_to_check:
                ranked = df[method].dropna().tolist()
                if top_k is not None:
                    ranked = ranked[:top_k]

                # ---- side imbalance metrics (computed on SAME ranked list) ----
                counts = self._zone_counts(ranked, zones)

                pair_imbal = {}
                pair_weights = {}
                for a, b in zone_pairs:
                    ia = counts.get(a, 0)
                    ib = counts.get(b, 0)
                    pair_imbal[f"imb_{a}_{b}"] = self._pair_imbalance(ia, ib)
                    pair_weights[f"w_{a}_{b}"] = ia + ib

                total_w = sum(pair_weights.values())
                overall_imb = (
                    sum(
                        pair_weights[k] * pair_imbal[k.replace("w_", "imb_")]
                        for k in pair_weights.keys()
                    ) / total_w
                ) if total_w > 0 else np.nan

                # ---- symmetry completion metrics ----
                feat_set = set(ranked)
                rank_pos = {f: i for i, f in enumerate(ranked)}

                seen_pairs = set()
                completed = []
                singletons = []

                for f in ranked:
                    sym = self._swap_prefix(f, swap_rules)
                    if sym is None:
                        continue

                    key = tuple(sorted([f, sym]))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    in_f = f in feat_set
                    in_s = sym in feat_set

                    if in_f and in_s:
                        completed.append((f, sym))
                    else:
                        existing = f if in_f else sym
                        missing = sym if in_f else f
                        singletons.append((existing, missing))

                denom = len(completed) + len(singletons)
                sym_score = (len(completed) / denom) if denom > 0 else np.nan

                out = {
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
    
    def ranking_agreement_table(self) -> pd.DataFrame:
        """
        Computes Kendall tau between the ORDERINGS in the top{comp_top} table:
        - base_col vs each comp_col (for each target)
        """
        rows = []
        for pred_obj in self.pred_objs:
            comp_top_name = self.comp_top_name_tpl.format(pred_obj=pred_obj)
            df = pd.read_csv(os.path.join(self.feat_path, comp_top_name))
            if self.base_col not in df.columns:
                raise ValueError(f"base_col '{self.base_col}' not found in {comp_top_name}")

            base_rank = df[self.base_col].dropna().tolist()
            base_pos = {f: i for i, f in enumerate(base_rank)}

            for col in self.comp_cols:
                if col not in df.columns:
                    continue
                other = df[col].dropna().tolist()
                # align to intersection so kendall makes sense
                aligned = [base_pos[f] for f in other if f in base_pos]
                tau = kendalltau(range(len(aligned)), aligned).statistic if len(aligned) >= 2 else np.nan

                rows.append({
                    "target": pred_obj,
                    "base_method": self.base_col,
                    "comp_method": col,
                    "kendall_tau": float(tau) if tau is not None else np.nan,
                    "n_intersection": int(len(aligned)),
                })

        return pd.DataFrame(rows)

    # ==================== Core: CV incremental curves (method comparison) ====================
    def incremental_curves_cv(self, max_add: Optional[int] = None) -> pd.DataFrame:
        """
        For each method in add_dict, evaluate:
        - baseline = top{base_top} base_col
        - add 1-by-1 following that method's ordering (features not in baseline)
        Output schema:
            target | method | fold | k_added | n_features | rmse | r2
        """
        # caching of results by max_add
        if self.cache_data:
            key = f"incremental_curves_{max_add}"
            if hasattr(self, "_cached_results") and key in self._cached_results:
                return self._cached_results[key]

        df = self._load_data_full()

        need = self.comp_top - self.base_top
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        rows = []
        for pred_obj in self._iter(self.pred_objs, desc="Targets (CV incremental)"):
            X, y = self._get_Xy(df, pred_obj)
            params = self._load_params(pred_obj)
            baseline, add_dict = self._ordered_features_from_table(pred_obj)

            # if max_add not specified: use the shortest add list (for comparability)
            max_add_use = max_add
            if max_add_use is None:
                max_add_use = min(len(v) for v in add_dict.values())

            splits = list(kf.split(X))

            for method, add_list in self._iter(add_dict.items(), desc=f"Methods for {pred_obj}", leave=False):
                add_list = add_list[:max_add_use]  # keep comparable length
                add_list = add_list[:need]  # also cap by comp_top - base_top to match table limits

                # parallelize across folds
                fold_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._process_fold)(X, y, params, baseline, add_list, method, fold, tr, te, max_add_use)
                    for fold, (tr, te) in enumerate(splits, start=1)
                )
                for fr in fold_results:
                    for r in fr:
                        r["target"] = pred_obj
                        rows.append(r)

        out = pd.DataFrame(rows)
        if self.cache_data:
            if not hasattr(self, "_cached_results"):
                self._cached_results = {}
            self._cached_results[key] = out
        return out

    # ==================== 2.3 Bootstrap empirical ranking stability (optional SHAP; default gain) ====================
    def bootstrap_empirical_ranking_stability(self, top_k: int = 25) -> pd.DataFrame:
        """
        Measures how *data-sensitive* an empirical ranking is under bootstrap resampling.
        Default: LightGBM gain importance as a proxy (fast, no extra deps).
        If include_shap_if_available=True and shap installed, uses SHAP mean(|shap|) ranking.

        Output per target:
            target | bootstrap_id | empirical_source | kendall_tau_vs_first | topk_overlap_vs_first
        """
        # caching by top_k and bootstrap_B
        if self.cache_data:
            key = f"bootstrap_{top_k}_{self.bootstrap_B}"
            if hasattr(self, "_cached_results") and key in self._cached_results:
                return self._cached_results[key]

        df = self._load_data_full()

        # Try SHAP if enabled
        shap_ok = False
        shap = None
        if self.include_shap_if_available:
            try:
                import shap as _shap  # noqa
                shap = _shap
                shap_ok = True
            except Exception:
                shap_ok = False

        rows = []
        for pred_obj in self._iter(self.pred_objs, desc="Targets (bootstrap ranking)"):
            X, y = self._get_Xy(df, pred_obj)
            params = self._load_params(pred_obj)
            rng = np.random.RandomState(self.random_state)

            def get_rank(Xb, yb) -> List[str]:
                booster = self._fit_booster_full(Xb, yb, params)
                if shap_ok:
                    explainer = shap.TreeExplainer(booster)
                    sv = explainer.shap_values(Xb, check_additivity=False)
                    imp = np.mean(np.abs(sv), axis=0)
                    return list(pd.Series(imp, index=Xb.columns).sort_values(ascending=False).index[:top_k])
                else:
                    imp = booster.feature_importance(importance_type=self.empirical_importance_type)
                    return list(pd.Series(imp, index=Xb.columns).sort_values(ascending=False).index[:top_k])

            # bootstrap ranks (parallel)
            def one_boot(b: int) -> List[str]:
                idx = rng.randint(0, len(X), size=len(X))
                Xb = X.iloc[idx]
                yb = y.iloc[idx]
                return get_rank(Xb, yb)

            ranks = Parallel(n_jobs=self.n_jobs)(
                delayed(one_boot)(b) for b in range(self.bootstrap_B)
            )

            ref = ranks[0]
            ref_pos = {f: i for i, f in enumerate(ref)}

            for b, rk in enumerate(ranks):
                inter = [ref_pos[f] for f in rk if f in ref_pos]
                tau = kendalltau(range(len(inter)), inter).statistic if len(inter) >= 2 else np.nan
                overlap = len(set(ref) & set(rk)) / float(top_k)

                rows.append({
                    "target": pred_obj,
                    "bootstrap_id": b,
                    "empirical_source": ("shap" if shap_ok else f"lgb_{self.empirical_importance_type}"),
                    "kendall_tau_vs_first": float(tau) if tau is not None else np.nan,
                    "topk_overlap_vs_first": float(overlap),
                })

        out = pd.DataFrame(rows)
        if self.cache_data:
            if not hasattr(self, "_cached_results"):
                self._cached_results = {}
            self._cached_results[key] = out
        return out

    # ==================== 2.4 Δk stability from CV curves ====================
    def delta_k_from_curve_cv(self, curve_df: pd.DataFrame, metric: str = "rmse") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Δk per fold for each method:
            Δk_fold = metric(k) - metric(k-1)
        Returns:
            delta_df: target|method|fold|n_features|delta_metric
            delta_summary: target|method|n_features|delta_mean|delta_std
        """
        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")

        d = curve_df.copy().sort_values(["target", "method", "fold", "n_features"])
        d[f"delta_{metric}"] = d.groupby(["target", "method", "fold"])[metric].diff()

        delta_df = d.dropna(subset=[f"delta_{metric}"]).copy()

        delta_summary = (
            delta_df.groupby(["target", "method", "n_features"])[f"delta_{metric}"]
            .agg(delta_mean="mean", delta_std="std")
            .reset_index()
            .sort_values(["target", "method", "n_features"])
        )

        return delta_df, delta_summary

    # ==================== 2.5 Sensitivity tests (noise + downsampling) ====================
    def sensitivity_curves_cv(self, max_add: Optional[int] = None, mode: str = "noise") -> pd.DataFrame:
        """
        Run incremental curves under controlled perturbations.

        mode="noise":
            for sigma in self.noise_sigmas:
                y' = y + N(0, sigma^2)   (applied to ALL samples consistently within a fold)
        mode="downsample":
            for frac in self.train_fracs:
                randomly sub-sample the TRAIN split only (keep test intact)

        Output schema:
            target | method | fold | n_features | k_added | rmse | r2 | perturb_type | perturb_value
        """
        if mode not in ["noise", "downsample"]:
            raise ValueError("mode must be 'noise' or 'downsample'")

        # caching by max_add and mode
        if self.cache_data:
            key = f"sensitivity_{mode}_{max_add}"
            if hasattr(self, "_cached_results") and key in self._cached_results:
                return self._cached_results[key]

        df = self._load_data_full()

        rows = []
        need = self.comp_top - self.base_top
        perturb_values = self.noise_sigmas if mode == "noise" else self.train_fracs
        perturb_type = "sigma" if mode == "noise" else "train_frac"

        for pred_obj in self._iter(self.pred_objs, desc="Targets (sensitivity)"):
            X, y = self._get_Xy(df, pred_obj)
            params = self._load_params(pred_obj)
            baseline, add_dict = self._ordered_features_from_table(pred_obj)

            # match the incremental_curves_cv adding rule:
            # - if max_add is None: use shortest add list for comparability
            # - always cap by (comp_top - base_top)
            max_add_use = max_add
            if max_add_use is None:
                max_add_use = min(len(v) for v in add_dict.values())
            max_add_use = min(max_add_use, need)

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))

            for pv in self._iter(perturb_values, desc=f"Perturbations ({mode}) for {pred_obj}", leave=False):
                for method, add_list in self._iter(add_dict.items(), desc="Methods", leave=False):
                    add_list = add_list[:max_add_use]
                    add_list = add_list[:need]

                    # parallelize across folds for each perturbation and method
                    def process_perturb_fold(fold: int, tr: np.ndarray, te: np.ndarray) -> List[Dict[str, Any]]:
                        X_tr_full, X_te = X.iloc[tr], X.iloc[te]
                        y_tr_full, y_te = y.iloc[tr], y.iloc[te]

                        # apply perturbation
                        if mode == "noise":
                            seed = int(self.random_state + fold * 1000 + int(pv * 1e6))
                            y_tr = self._apply_target_noise(y_tr_full, sigma=float(pv), seed=seed)
                            y_te_use = self._apply_target_noise(y_te, sigma=float(pv), seed=seed + 1)
                            X_tr = X_tr_full
                        else:
                            rng = np.random.RandomState(self.random_state + fold * 1000 + int(pv * 100))
                            n_tr = len(X_tr_full)
                            keep = rng.choice(n_tr, size=max(5, int(n_tr * float(pv))), replace=False)
                            X_tr = X_tr_full.iloc[keep]
                            y_tr = y_tr_full.iloc[keep]
                            y_te_use = y_te  # unchanged

                        subrows = []
                        for k_added, feat_list in self._incremental_feature_sets(baseline, add_list, max_add=max_add_use):
                            m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te_use, params, feat_list)
                            subrows.append({
                                "target": pred_obj,
                                "method": method,
                                "fold": fold,
                                "k_added": k_added,
                                "n_features": len(feat_list),
                                "rmse": m["rmse"],
                                "r2": m["r2"],
                                "perturb_type": perturb_type,
                                "perturb_value": float(pv),
                            })
                        return subrows

                    fold_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(process_perturb_fold)(fold, tr, te)
                        for fold, (tr, te) in enumerate(splits, start=1)
                    )
                    for fr in fold_results:
                        rows.extend(fr)

        out = pd.DataFrame(rows)
        if self.cache_data:
            if not hasattr(self, "_cached_results"):
                self._cached_results = {}
            self._cached_results[key] = out
        return out
    
    def summarize_sensitivity_curves(self, sens_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize mean/std across folds for each method, perturbation, and n_features.
        """
        return (
            sens_df.groupby(["target", "method", "perturb_type", "perturb_value", "n_features"])
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
            )
            .reset_index()
            .sort_values(["perturb_value", "method", "n_features"])
        )
    
    # ==================== Export all Task2 outputs into one Excel ====================
    def export_to_excel(self, out_xlsx_path: str, max_add: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        # 2.2 ranking agreement (from table)
        agree_df = self.ranking_agreement_table()
        sym_df = self.symmetry_bias_table(swap_rules=[("ll_", "rr_"), ("lf_", "rf_"), ("ff_", "bb_"), ("lb_", "rb_")], top_k=self.comp_top, rank_weighted=True)
        ped_cont_df = self.ped_continuity_table(top_k=self.comp_top)
        # core CV curves (for 2.1 and 2.4 too)
        curve_df = self.incremental_curves_cv(max_add=max_add)

        # 2.1 effective snr (from terminal R2)
        eff_snr_df = self.effective_snr_by_method_cv(curve_df)

        # 2.4 delta-k (rmse / r2)
        delta_rmse_df, delta_rmse_summary = self.delta_k_from_curve_cv(curve_df, metric="rmse")
        delta_r2_df, delta_r2_summary = self.delta_k_from_curve_cv(curve_df, metric="r2")

        # 2.3 bootstrap empirical ranking stability
        boot_rank_df = self.bootstrap_empirical_ranking_stability(top_k=self.comp_top)

        # 2.5 sensitivity: noise
        sens_noise_df = self.sensitivity_curves_cv(max_add=max_add, mode="noise")
        sens_noise_summary = self.summarize_sensitivity_curves(sens_noise_df)

        # 2.5 sensitivity: downsample
        sens_down_df = self.sensitivity_curves_cv(max_add=max_add, mode="downsample")
        sens_down_summary = self.summarize_sensitivity_curves(sens_down_df)

        # mean curve table for convenience
        mean_curve_df = (
            curve_df.groupby(["target", "method", "n_features"])
            .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                 r2_mean=("r2", "mean"), r2_std=("r2", "std"))
            .reset_index()
            .sort_values(["target", "method", "n_features"])
        )

        with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
            agree_df.to_excel(writer, sheet_name="ranking_agreement", index=False)
            sym_df.to_excel(writer, sheet_name="symmetry_bias", index=False)
            ped_cont_df.to_excel(writer, sheet_name="locality_bias", index=False)
            curve_df.to_excel(writer, sheet_name="curves_cv_all", index=False)
            mean_curve_df.to_excel(writer, sheet_name="curves_cv_mean", index=False)
            eff_snr_df.to_excel(writer, sheet_name="effective_snr", index=False)

            delta_rmse_df.to_excel(writer, sheet_name="delta_rmse_per_fold", index=False)
            delta_rmse_summary.to_excel(writer, sheet_name="delta_rmse_summary", index=False)
            delta_r2_df.to_excel(writer, sheet_name="delta_r2_per_fold", index=False)
            delta_r2_summary.to_excel(writer, sheet_name="delta_r2_summary", index=False)

            boot_rank_df.to_excel(writer, sheet_name="bootstrap_rank_stab", index=False)
            sens_noise_df.to_excel(writer, sheet_name="noise_curves", index=False)
            sens_noise_summary.to_excel(writer, sheet_name="noise_summary", index=False)
            sens_down_df.to_excel(writer, sheet_name="down_curves", index=False)
            sens_down_summary.to_excel(writer, sheet_name="down_summary", index=False)

        return {
            "curve_df": curve_df,
            "mean_curve_df": mean_curve_df,
            "eff_snr_df": eff_snr_df,
            "delta_rmse_df": delta_rmse_df,
            "delta_rmse_summary": delta_rmse_summary,
            "delta_r2_df": delta_r2_df,
            "delta_r2_summary": delta_r2_summary,
            "boot_rank_df": boot_rank_df,
            "sens_noise_df": sens_noise_df,
            "sens_noise_summary": sens_noise_summary,
            "sens_down_df": sens_down_df,
            "sens_down_summary": sens_down_summary,
        }

#%% run code
if __name__ == "__main__":
    sm = SelectionMechanismAnalyzer(
        data_path="../data/",
        model_path="../output_full/",
        feat_path="../output_single_featimp/",
        flow_type="entrance",
        pred_objs=["step_x","step_y"],
        base_top=5,
        comp_top=25,
        base_col="top_shap",
        comp_cols=["llm_selection_wo_val", "fused_importance"],
        n_splits=5,
        bootstrap_B=5,
        empirical_importance_type="gain",
        include_shap_if_available=False,   # set True only if you installed shap
        noise_sigmas =[0.1, 0.25],
        train_fracs = [0.9, 0.75],
        # speed options
        n_jobs=4,                  # make use of multiple cores
        show_progress=True,        # display tqdm bars
        cache_data=True,           # reuse intermediate results on reruns
    )

    out_xlsx = os.path.join(sm.feat_path, f"SelectionMechanismComp_{sm.flow_type}.xlsx")
    print(out_xlsx)
    export_outputs = sm.export_to_excel(out_xlsx, max_add=20)
    print(f"Saved: {out_xlsx}")

    # quick plots (optional)
    df_curve = export_outputs["curve_df"]
    sm.plot_mean_curve_with_band(df_curve, metric="rmse")
    delta_rmse_summary = export_outputs["delta_rmse_summary"]
    sm.plot_delta_k_with_band(delta_rmse_summary, metric="rmse")
    sens_noise_summary = export_outputs["sens_noise_summary"]
    sm.plot_sensitivity_mean_curves(sens_noise_summary, metric="rmse", max_panels=4)
# %%