#%% import libraries
import os
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

#%%% define data class for use
@dataclass
class SignalRegimeAnalyzer:
    data_path: str
    model_path: str
    feat_path: str
    flow_type: str
    pred_objs: List[str]                 
    base_top: int                        
    comp_top: int                        
    base_col: str                       
    n_splits: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        self.file_name = f"ds_vector_{self.flow_type}.csv"
        self.base_top_name_tpl = f"{self.flow_type}" + "_{pred_obj}_candidate_features_top" + str(self.base_top) + ".csv"
        self.comp_top_name_tpl = f"{self.flow_type}" + "_{pred_obj}_candidate_features_top" + str(self.comp_top) + ".csv"
        
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
    
# ---------- shared IO ----------
    def _load_data_full(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df = df.drop(columns = ["id", "time"], errors = "ignore")
        return df

    def _load_params(self, pred_obj: str) -> dict:
        param_name = f"{self.flow_type}_{pred_obj}_lgb_param.joblib"
        params = joblib.load(os.path.join(self.model_path, param_name))
        if not isinstance(params, dict):
            raise ValueError("Loaded parameters are not in dictionary format.")
        return params

    def _get_Xy(self, df: pd.DataFrame, pred_obj: str) -> Tuple[pd.DataFrame, pd.Series]:
        y = df[pred_obj]
        X = df.drop(columns=["step_x", "step_y"], errors="ignore")
        return X, y

    # ---------- model eval (match your style: lightgbm.train) ----------
    def _train_eval_lgb(self, X_train, y_train, X_test, y_test, params, feature_list,
                        early_stopping_rounds=100, verbose_eval=0) -> Dict[str, float]:

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

    # ---------- Task 1.1 distribution ----------
    def target_distribution_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for t in self.pred_objs:
            y = df[t].values
            rows.append({
                "target": t,
                "variance": float(np.var(y)),
                "skewness": float(skew(y)),
                "kurtosis": float(kurtosis(y)),
            })
        return pd.DataFrame(rows)

    # ---------- Task 1.2 target-level SNR (full feature model, CV) ----------
    def target_level_snr_cv(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for t in self.pred_objs:
            X, y = self._get_Xy(df, t)
            params = self._load_params(t)

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            r2s = []
            rmses = []

            feat_list = list(X.columns)

            for fold, (tr, te) in enumerate(kf.split(X), start=1):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]
                m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te, params, feat_list)
                r2s.append(m["r2"])
                rmses.append(m["rmse"])

            mean_r2 = float(np.mean(r2s))
            snr = float(mean_r2 / (1 - mean_r2)) if mean_r2 < 1 else np.inf

            rows.append({
                "target": t,
                "cv_r2_mean": mean_r2,
                "cv_r2_std": float(np.std(r2s)),
                "cv_rmse_mean": float(np.mean(rmses)),
                "cv_rmse_std": float(np.std(rmses)),
                "snr_estimate_r2_over_1_minus_r2": snr,
            })

        return pd.DataFrame(rows)

    # ---------- Task 1.3 redundancy metrics ----------
    def redundancy_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        # Same X for both targets (your data uses same inputs), so compute once.
        X = df.drop(columns=["step_x", "step_y"], errors="ignore")
        corr = X.corr().abs()
        mean_abs_corr = float(corr.values[np.triu_indices_from(corr, k=1)].mean())
        cond_number = float(np.linalg.cond(X.values))

        for t in self.pred_objs:
            rows.append({
                "target": t,
                "mean_abs_corr": mean_abs_corr,
                "condition_number": cond_number,
            })
        return pd.DataFrame(rows)

    # ---------- helper: read feature lists (follow your naming scheme) ----------
    def _load_feature_lists(self, pred_obj: str) -> Tuple[List[str], List[str]]:
        base_top_name = self.base_top_name_tpl.format(pred_obj=pred_obj)
        comp_top_name = self.comp_top_name_tpl.format(pred_obj=pred_obj)

        df_base_top = pd.read_csv(os.path.join(self.feat_path, base_top_name))
        df_comp_top = pd.read_csv(os.path.join(self.feat_path, comp_top_name))

        start_base = df_base_top[self.base_col].dropna().tolist()
        end_base = df_comp_top[self.base_col].dropna().tolist()
        add_feats = [f for f in end_base if f not in start_base]

        return start_base, add_feats

    def _incremental_feature_sets(self, base: List[str], to_add_in_order: List[str], max_add: Optional[int]):
        if max_add is None:
            max_add = len(to_add_in_order)
        for k in range(0, max_add + 1):
            yield k, base + to_add_in_order[:k]

    # ---------- Task 1.4 incremental curve stability (CV) ----------
    def incremental_curve_cv(self, df: pd.DataFrame, max_add: int) -> pd.DataFrame:
        """
        Returns per-fold curve:
        target, fold, k_added, n_features, rmse, r2
        Baseline = top{base_top} of base_col. Then add base_col features up to top{comp_top}.
        """
        rows = []
        for t in tqdm(self.pred_objs, desc = "Targets (incremental CV)"):
            X, y = self._get_Xy(df, t)
            params = self._load_params(t)

            base_feats, add_feats = self._load_feature_lists(t)

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))
            for fold, (tr, te) in tqdm(list(enumerate(splits, start=1)), desc = f"Folds for {t}", leave = False):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]

                for k_added, feat_list in tqdm(list(self._incremental_feature_sets(base_feats, add_feats, max_add=max_add)), desc = f"k-add for {t} fold {fold}", leave=False):
                    m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te, params, feat_list)
                    rows.append({
                        "target": t,
                        "fold": fold,
                        "k_added": k_added,
                        "n_features": len(feat_list),
                        "rmse": m["rmse"],
                        "r2": m["r2"],
                    })

        curve_df = pd.DataFrame(rows)
        return curve_df
    
    def averaging_from_curve(self, curve_df: pd.DataFrame) -> pd.DataFrame:
        mean_curve_df = (curve_df.groupby(["target", "n_features"]).agg(rmse_mean=("rmse", "mean"), 
                                                                        rmse_std=("rmse", "std"), 
                                                                        r2_mean=("r2", "mean"), 
                                                                        r2_std=("r2", "std"))
        .reset_index()
        .sort_values(["target", "n_features"]))
        return mean_curve_df

    def incremental_stability_from_curve(self, curve_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes stability metrics from CV curves:
        - mean/std RMSE per n_features
        - Δk stats computed on the MEAN curve (wiggliness)
        - fold-level Δk std (instability across folds)
        """
        out_rows = []

        for t in self.pred_objs:
            d = curve_df[curve_df["target"] == t].copy()

            # aggregate mean curve
            agg = d.groupby("n_features").agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
            ).reset_index().sort_values("n_features")

            # Δk on mean curve (wiggliness across k)
            agg["delta_rmse_mean_curve"] = agg["rmse_mean"].diff()
            deltas = agg["delta_rmse_mean_curve"].dropna().values

            std_over_k = float(np.std(deltas)) if len(deltas) else np.nan

            # flip ratio on mean curve
            if len(deltas) >= 2:
                s = np.sign(deltas)
                flip_ratio = float(np.mean(s[1:] != s[:-1]))
            else:
                flip_ratio = np.nan

            # fold-level Δk variability:
            d = d.sort_values(["fold", "n_features"])
            d["delta_rmse_fold"] = d.groupby("fold")["rmse"].diff()

            fold_std_per_k = (
                d.dropna()
                 .groupby("n_features")["delta_rmse_fold"]
                 .std()
                 .mean()
            )
            fold_std_per_k = float(fold_std_per_k) if not np.isnan(fold_std_per_k) else np.nan

            out_rows.append({
                "target": t,
                "std_over_k_mean_curve": std_over_k,
                "flip_ratio_mean_curve": flip_ratio,
                "mean_std_delta_rmse_across_folds": fold_std_per_k,
            })

        return pd.DataFrame(out_rows)
    
    # ---------- Task 1.5 nonlinearity dependence (CV) ----------
    def nonlinearity_gap_cv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantify interaction/nonlinearity reliance by comparing LGB vs Linear:
        gap = R2_lgb - R2_linear (averaged across folds).
        """
        rows = []
        for t in self.pred_objs:
            X, y = self._get_Xy(df, t)
            params = self._load_params(t)

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            gaps = []
            r2_lins = []
            r2_lgbs = []

            feat_list = list(X.columns)

            for fold, (tr, te) in enumerate(kf.split(X), start=1):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]

                # Linear baseline
                lin = LinearRegression()
                lin.fit(X_tr, y_tr)
                y_pred_lin = lin.predict(X_te)
                r2_lin = float(r2_score(y_te, y_pred_lin))

                # LGB using same optimized params (all features)
                m = self._train_eval_lgb(X_tr, y_tr, X_te, y_te, params, feat_list)
                r2_lgb = float(m["r2"])

                r2_lins.append(r2_lin)
                r2_lgbs.append(r2_lgb)
                gaps.append(r2_lgb - r2_lin)

            rows.append({
                "target": t,
                "r2_linear_mean": float(np.mean(r2_lins)),
                "r2_linear_std": float(np.std(r2_lins)),
                "r2_lgb_mean": float(np.mean(r2_lgbs)),
                "r2_lgb_std": float(np.std(r2_lgbs)),
                "nonlinearity_gap_mean": float(np.mean(gaps)),
                "nonlinearity_gap_std": float(np.std(gaps)),
            })

        return pd.DataFrame(rows)

    # ---------- Export everything into one Excel workbook ----------
    def export_all_to_excel(self, max_add: int = 20):
        df = self._load_data_full()

        # compute
        dist_df = self.target_distribution_metrics(df)
        snr_df = self.target_level_snr_cv(df)
        red_df = self.redundancy_metrics(df)
        nonlin_df = self.nonlinearity_gap_cv(df)
        curve_df = self.incremental_curve_cv(df, max_add=max_add)
        mean_curve_df = self.averaging_from_curve(curve_df)
        stability_df = self.incremental_stability_from_curve(curve_df)
        delta_rmse_df, delta_rmse_summary = self.delta_k_from_curve_cv(curve_df, metric="rmse")
        delta_r2_df, delta_r2_summary = self.delta_k_from_curve_cv(curve_df, metric="r2")

        # write workbook with multiple tabs (each tab = target metric)
        out_xlsx_path = os.path.join(self.feat_path, f'{self.flow_type}_Statistics.xlsx')
        with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
            dist_df.to_excel(writer, sheet_name="distribution", index=False)
            snr_df.to_excel(writer, sheet_name="snr_cv", index=False)
            red_df.to_excel(writer, sheet_name="redundancy", index=False)
            nonlin_df.to_excel(writer, sheet_name="nonlinearity_gap", index=False)
            curve_df.to_excel(writer, sheet_name="incremental_curve_cv", index=False)
            mean_curve_df.to_excel(writer, sheet_name="incremental_mean_curve", index=False)
            stability_df.to_excel(writer, sheet_name="incremental_stability", index=False)
            delta_rmse_summary.to_excel(writer, sheet_name="delta_rmse_summary", index=False)
            delta_r2_summary.to_excel(writer, sheet_name="delta_r2_summary", index=False)
        return mean_curve_df, delta_rmse_summary
        
    # ---------- Plot mean incremental curve with std band ----------
    def plot_mean_incremental_curve(self, mean_curve_df: pd.DataFrame, metric: str = "rmse", save: bool = True):
        """
        Plot CV mean incremental curve with ±1 std band.
        mean_curve_df must contain:
        target | n_features | rmse_mean | rmse_std | r2_mean | r2_std
        """

        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        for t in self.pred_objs:
            df_plot = (mean_curve_df[mean_curve_df["target"] == t].sort_values("n_features").reset_index(drop=True))

            x = df_plot["n_features"]
            y = df_plot[mean_col]
            y_std = df_plot[std_col]

            plt.figure(figsize=(9, 6))
            plt.plot(x, y, marker="o", linewidth=2, label=f"{t} mean {metric}")
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.25)

            plt.title(f"{self.flow_type} | {t} | CV Mean {metric} vs n_features")
            plt.xlabel("n_features")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # force x-axis major ticks every 1
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

            if save:
                save_name = f"{self.flow_type}_{t}_CVMean_{metric}_Incremental_{self.base_top}_{self.comp_top}.png"
                save_path = os.path.join(self.feat_path, save_name)
                plt.tight_layout()
                plt.savefig(save_path, dpi=400)

            plt.show()

    def delta_k_from_curve_cv(self, curve_df: pd.DataFrame, metric: str = "rmse") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute Δk per fold:
            Δk_fold = metric(k) - metric(k-1)
        Returns:
            delta_df: target|fold|n_features|delta_metric
            delta_summary: target|n_features|delta_mean|delta_std
        """
        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")

        d = curve_df.copy().sort_values(["target", "fold", "n_features"])
        d[f"delta_{metric}"] = d.groupby(["target", "fold"])[metric].diff()

        delta_df = d.dropna(subset=[f"delta_{metric}"]).copy()

        delta_summary = (
            delta_df.groupby(["target", "n_features"])[f"delta_{metric}"]
            .agg(delta_mean="mean", delta_std="std")
            .reset_index()
            .sort_values(["target", "n_features"])
        )

        return delta_df, delta_summary 

    # ---------- Plot delta-k marginal gain curve with approx std band ----------
    def plot_delta_k_marginal_from_cv(self, delta_summary_df: pd.DataFrame, metric: str = "rmse", save: bool = True):
        """
        Plot fold-aggregated Δk:
            mean ± std across folds
        delta_summary_df must contain:
            target | n_features | delta_mean | delta_std
        """
        if metric not in ["rmse", "r2"]:
            raise ValueError("metric must be 'rmse' or 'r2'")

        for t in self.pred_objs:
            d = (delta_summary_df[delta_summary_df["target"] == t].sort_values("n_features").reset_index(drop=True))
            x = d["n_features"]
            y = d["delta_mean"]
            y_std = d["delta_std"].fillna(0)

            plt.figure(figsize=(9, 6))
            plt.plot(x, y, marker="o", linewidth=2, label=f"{t} Δ{metric} (mean)")
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.25)
            plt.axhline(0, linewidth=1, alpha=0.6)

            plt.title(f"{self.flow_type} | {t} | Δ{metric} (CV mean±std) vs n_features")
            plt.xlabel("n_features (k)")
            plt.ylabel(f"Δ{metric} = {metric}(k) - {metric}(k-1)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            # force x-axis major ticks every 1
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            
            if save:
                save_name = f"{self.flow_type}_{t}_CV_Delta_{metric}_MeanStd.png"
                save_path = os.path.join(self.feat_path, save_name)
                plt.tight_layout()
                plt.savefig(save_path, dpi=400)

            plt.show()

#%% run code
if __name__ == "__main__":
    srt = SignalRegimeAnalyzer(
        data_path="../data/",
        model_path="../output_full/",
        feat_path="../output_single_featimp/",
        flow_type="entrance",
        pred_objs=["step_x", "step_y"],
        base_top = 5,
        comp_top = 25,
        base_col = "top_shap",
        n_splits = 5,
    )
    mean_curve_df, delta_rmse_summary = srt.export_all_to_excel(max_add=20)
    srt.plot_mean_incremental_curve(mean_curve_df, metric = 'rmse')
    srt.plot_delta_k_marginal_from_cv(delta_rmse_summary, metric="rmse")

# %%
