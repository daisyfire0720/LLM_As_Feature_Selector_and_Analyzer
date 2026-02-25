
#%%% import packages 
import os
from fastavro import writer
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import lightgbm as lgb

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

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

    noise_sigmas: Optional[List[float]] = None          # e.g. [0.0, 0.1, 0.2]
    train_fracs: Optional[List[float]] = None           # e.g. [1.0, 0.7, 0.4]

    def __post_init__(self):
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
    
    def _normalize_method_name(self, method: str) -> str:
        """Normalize method names for consistent display.
        Converts any llm_selection_* variant to just 'llm_selection'.
        """
        if "llm_selection" in method:
            return "llm_selection"
        return method
    
    def _get_method_color(self, method: str) -> str:
        """Get color for a given method, using normalized name."""
        normalized = self._normalize_method_name(method)
        return self.method_colors.get(normalized, "#000000")  # default to black if not found
    
    # -------------------- shared IO --------------------
    def _load_data_full(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df = df.drop(columns=["id", "time"], errors="ignore")
        return df

    def _get_Xy(self, df: pd.DataFrame, pred_obj: str) -> Tuple[pd.DataFrame, pd.Series]:
        y = df[pred_obj]
        X = df.drop(columns=["step_x", "step_y"], errors="ignore")
        return X, y

    def _load_params(self, pred_obj: str) -> dict:
        param_name = f"{self.flow_type}_{pred_obj}_lgb_param.joblib"
        params = joblib.load(os.path.join(self.model_path, param_name))
        if not isinstance(params, dict):
            raise ValueError("Loaded parameters are not in dictionary format.")
        return params

    # -------------------- LGB train/eval (match your style: lgb.train) --------------------
    def _train_eval_lgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: dict,
        feature_list: List[str],
        early_stopping_rounds: int = 100,
        verbose_eval: int = 0,
    ) -> Dict[str, float]:

        dtrain = lgb.Dataset(X_train[feature_list], label=y_train, free_raw_data=False)
        dvalid = lgb.Dataset(X_test[feature_list], label=y_test, reference=dtrain, free_raw_data=False)

        p = params.copy()
        p["verbosity"] = -1
        p["device"] = "gpu"

        model = lgb.train(
            params=p,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=bool(verbose_eval)),
                lgb.log_evaluation(period=verbose_eval) if verbose_eval else lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_test[feature_list], num_iteration=model.best_iteration)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        return {"rmse": rmse, "r2": r2, "best_iteration": int(model.best_iteration)}

    # -------------------- feature list extraction (your baseline logic) --------------------
    def _ordered_features_from_table(self, pred_obj: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        baseline = top{base_top} from base_col (common features)
        For each method col, build add_list = features in top{comp_top} of that col but not in baseline.
        """
        base_top_name = self.base_top_name_tpl.format(pred_obj=pred_obj)
        comp_top_name = self.comp_top_name_tpl.format(pred_obj=pred_obj)
        df_base = pd.read_csv(os.path.join(self.feat_path, base_top_name))
        df_comp = pd.read_csv(os.path.join(self.feat_path, comp_top_name))

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

    def _incremental_feature_sets(self, base: List[str], to_add_in_order: List[str], max_add: Optional[int]):
        if max_add is None:
            max_add = len(to_add_in_order)
        for k in range(0, max_add + 1):
            yield k, base + to_add_in_order[:k]

    # --------------------- apply perturbations helper function -------------------
    def _apply_target_noise(self, y: pd.Series, sigma: float, seed: int) -> pd.Series:
        if sigma == 0.0:
            return y
        rng = np.random.RandomState(seed)
        return y + rng.normal(loc=0.0, scale=sigma, size=len(y))

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

    # ==================== 2.2 Ranking agreement (SHAP vs LLM vs fused) ====================
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
        df = self._load_data_full()

        need = self.comp_top - self.base_top
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        rows = []
        for pred_obj in tqdm(self.pred_objs, desc="Targets (CV incremental)"):
            X, y = self._get_Xy(df, pred_obj)
            params = self._load_params(pred_obj)
            baseline, add_dict = self._ordered_features_from_table(pred_obj)

            # if max_add not specified: use the shortest add list (for comparability)
            max_add_use = max_add
            if max_add_use is None:
                max_add_use = min(len(v) for v in add_dict.values())

            splits = list(kf.split(X))

            for method, add_list in tqdm(add_dict.items(), desc=f"Methods for {pred_obj}", leave=False):
                add_list = add_list[:max_add_use]  # keep comparable length
                add_list = add_list[:need]  # also cap by comp_top - base_top to match table limits

                for fold, (tr, te) in tqdm(list(enumerate(splits, start=1)), desc=f"Folds for {method}", leave=False):
                    X_tr, X_te = X.iloc[tr], X.iloc[te]
                    y_tr, y_te = y.iloc[tr], y.iloc[te]

                    for k_added, feat_list in tqdm(
                        list(self._incremental_feature_sets(baseline, add_list, max_add=max_add_use)),
                        desc=f"k-add {method} fold {fold}", leave=False
                    ):
                        m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te, params, feat_list)
                        rows.append({
                            "target": pred_obj,
                            "method": method,
                            "fold": fold,
                            "k_added": k_added,
                            "n_features": len(feat_list),
                            "rmse": m["rmse"],
                            "r2": m["r2"],
                        })

        return pd.DataFrame(rows)

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

    # ==================== 2.3 Bootstrap empirical ranking stability (optional SHAP; default gain) ====================
    def _fit_booster_full(self, X_train, y_train, params) -> lgb.Booster:
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        p = params.copy()
        p["verbosity"] = -1
        booster = lgb.train(params=p, train_set=dtrain)
        return booster

    def bootstrap_empirical_ranking_stability(self, top_k: int = 25) -> pd.DataFrame:
        """
        Measures how *data-sensitive* an empirical ranking is under bootstrap resampling.
        Default: LightGBM gain importance as a proxy (fast, no extra deps).
        If include_shap_if_available=True and shap installed, uses SHAP mean(|shap|) ranking.

        Output per target:
            target | bootstrap_id | empirical_source | kendall_tau_vs_first | topk_overlap_vs_first
        """
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
        for pred_obj in tqdm(self.pred_objs, desc="Targets (bootstrap ranking)"):
            X, y = self._get_Xy(df, pred_obj)
            params = self._load_params(pred_obj)
            rng = np.random.RandomState(self.random_state)

            def get_rank(Xb, yb) -> List[str]:
                booster = self._fit_booster_full(Xb, yb, params)
                if shap_ok:
                    # TreeExplainer for LGB Booster
                    explainer = shap.TreeExplainer(booster)
                    sv = explainer.shap_values(Xb, check_additivity=False)
                    # sv can be list for multiclass; here regression -> ndarray
                    imp = np.mean(np.abs(sv), axis=0)
                    return list(pd.Series(imp, index=Xb.columns).sort_values(ascending=False).index[:top_k])
                else:
                    imp = booster.feature_importance(importance_type=self.empirical_importance_type)
                    return list(pd.Series(imp, index=Xb.columns).sort_values(ascending=False).index[:top_k])

            # bootstrap ranks
            ranks = []
            for b in tqdm(range(self.bootstrap_B), desc=f"Bootstrap for {pred_obj}", leave=False):
                idx = rng.randint(0, len(X), size=len(X))
                Xb = X.iloc[idx]
                yb = y.iloc[idx]
                ranks.append(get_rank(Xb, yb))

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

        return pd.DataFrame(rows)

    # ==================== plotting (mean curves + delta-k) ====================
    def plot_mean_curve_with_band(self, curve_df: pd.DataFrame, metric: str = "rmse", save: bool = True):
        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")

        agg = (
            curve_df.groupby(["method", "n_features"])
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

        target = curve_df["target"].iloc[0] if "target" in curve_df.columns else self.pred_objs[0]
        plt.title(f"{self.flow_type} | {target} | CV Mean {metric} vs n_features (methods)")
        plt.xlabel("n_features")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save:
            target = curve_df["target"].iloc[0] if "target" in curve_df.columns else self.pred_objs[0]
            name = f"{self.flow_type}_{target}_Task2_CVMean_{metric}_Methods.png"
            path = os.path.join(self.feat_path, name)
            plt.tight_layout()
            plt.savefig(path, dpi=400)

        # force x-axis major ticks every 1
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.show()

    def plot_delta_k_with_band(self, delta_summary_df: pd.DataFrame, metric: str = "rmse", save: bool = True):
        plt.figure(figsize=(10, 6))

        for method in sorted(delta_summary_df["method"].unique()):
            d = delta_summary_df[delta_summary_df["method"] == method].sort_values("n_features")
            x = d["n_features"].values
            y = d["delta_mean"].values
            s = np.nan_to_num(d["delta_std"].values, nan=0.0)
            
            # Use normalized name for label and method color
            label = self._normalize_method_name(method)
            color = self._get_method_color(method)

            plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
            plt.fill_between(x, y - s, y + s, alpha=0.20, color=color)

        plt.axhline(0, linewidth=1, alpha=0.6)
        target = delta_summary_df["target"].iloc[0] if "target" in delta_summary_df.columns else self.pred_objs[0]
        plt.title(f"{self.flow_type} | {target} | Δ{metric} (CV mean±std) vs n_features (methods)")
        plt.xlabel("n_features (k)")
        plt.ylabel(f"Δ{metric} = {metric}(k) - {metric}(k-1)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save:
            target = delta_summary_df["target"].iloc[0] if "target" in delta_summary_df.columns else self.pred_objs[0]
            name = f"{self.flow_type}_{target}_Task2_CV_Delta_{metric}_Methods.png"
            path = os.path.join(self.feat_path, name)
            plt.tight_layout()
            plt.savefig(path, dpi=400)

        # force x-axis major ticks every 1
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.show()

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

        df = self._load_data_full()

        rows = []
        perturb_values = self.noise_sigmas if mode == "noise" else self.train_fracs
        perturb_type = "sigma" if mode == "noise" else "train_frac"

        for pred_obj in tqdm(self.pred_objs, desc="Targets (sensitivity)"):
            X, y = self._get_Xy(df, pred_obj)
            params = self._load_params(pred_obj)
            baseline, add_dict = self._ordered_features_from_table(pred_obj)

            if max_add is None:
                max_add = min(len(v) for v in add_dict.values())

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))

            for pv in tqdm(perturb_values, desc=f"Perturbations ({mode}) for {pred_obj}", leave=False):
                for method, add_list in tqdm(add_dict.items(), desc="Methods", leave=False):
                    add_list = add_list[:max_add]

                    for fold, (tr, te) in tqdm(list(enumerate(splits, start=1)), desc=f"Folds {method}", leave=False):
                        X_tr_full, X_te = X.iloc[tr], X.iloc[te]
                        y_tr_full, y_te = y.iloc[tr], y.iloc[te]

                        # apply perturbation
                        if mode == "noise":
                            # make noise reproducible per (pv, fold)
                            seed = int(self.random_state + fold * 1000 + int(pv * 1e6))
                            y_tr = self._apply_target_noise(y_tr_full, sigma=float(pv), seed=seed)
                            y_te_use = self._apply_target_noise(y_te, sigma=float(pv), seed=seed + 1)
                            X_tr = X_tr_full
                        else:
                            # downsample train only
                            rng = np.random.RandomState(self.random_state + fold * 1000 + int(pv * 100))
                            n_tr = len(X_tr_full)
                            keep = rng.choice(n_tr, size=max(5, int(n_tr * float(pv))), replace=False)
                            X_tr = X_tr_full.iloc[keep]
                            y_tr = y_tr_full.iloc[keep]
                            y_te_use = y_te  # unchanged

                        for k_added, feat_list in self._incremental_feature_sets(baseline, add_list, max_add=max_add):
                            m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te_use, params, feat_list)
                            rows.append({
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

        return pd.DataFrame(rows)
    
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
    
    def plot_sensitivity_mean_curves(
        self,
        sens_summary_df: pd.DataFrame,
        metric: str = "rmse",
        save: bool = True,
        max_panels: int = 6
    ):
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
                    name = f"{self.flow_type}_{target}_Task2_5_{metric}_{perturb_type}_{pv}.png"
                    path = os.path.join(self.feat_path, name)
                    plt.tight_layout()
                    plt.savefig(path, dpi=400)

                plt.show()
    
    # ==================== Export all Task2 outputs into one Excel ====================
    def export_to_excel(self, out_xlsx_path: str, max_add: Optional[int] = None):
        # 2.2 ranking agreement (from table)
        agree_df = self.ranking_agreement_table()

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

#%% run code
if __name__ == "__main__":
    sm = SelectionMechanismAnalyzer(
        data_path="../data/",
        model_path="../output_full/",
        feat_path="../output_single_featimp/",
        flow_type="corner",
        pred_objs=["step_x", "step_y"],
        base_top=5,
        comp_top=25,
        base_col="top_shap",
        comp_cols=["llm_selection_wo_val", "fused_importance"],
        n_splits=5,
        bootstrap_B=10,
        empirical_importance_type="gain",
        include_shap_if_available=False,   # set True only if you installed shap
        noise_sigmas=[0.1, 0.3],
        train_fracs=[0.7, 0.5],
    )

    out_xlsx = os.path.join(sm.feat_path, f"SelectionMechanismComp_{sm.flow_type}_all_targets.xlsx")
    print(out_xlsx)
    sm.export_to_excel(out_xlsx, max_add=20)
    print(f"Saved: {out_xlsx}")

    # quick plots (optional)
    df_curve = sm.incremental_curves_cv(max_add=20)
    sm.plot_mean_curve_with_band(df_curve, metric="rmse")
    _, delta_rmse_summary = sm.delta_k_from_curve_cv(df_curve, metric="rmse")
    sm.plot_delta_k_with_band(delta_rmse_summary, metric="rmse")
    sens_noise_df = sm.sensitivity_curves_cv(max_add=20, mode="noise")
    sens_noise_summary = sm.summarize_sensitivity_curves(sens_noise_df)
    sm.plot_sensitivity_mean_curves(sens_noise_summary, metric="rmse", max_panels=4)
# %%
