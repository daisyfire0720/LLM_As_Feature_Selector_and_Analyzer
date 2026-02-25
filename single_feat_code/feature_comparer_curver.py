#%%% import libraries
import os
import traceback
import json
import re
import joblib
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import lightgbm as lgb
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

#%% define class

@dataclass
class PerformanceComparer:
    data_path: str
    model_path: str
    feat_path: str
    flow_type: str
    pred_obj: str
    base_top: int
    comp_top: int
    base_col: str
    comp_col: List[str] | str
    
    def __post_init__(self):
        self.file_name = f"ds_vector_{self.flow_type}.csv"
        self.param_name = f'{self.flow_type}_{self.pred_obj}_lgb_param.joblib'
        self.base_top_name = f'{self.flow_type}_{self.pred_obj}_candidate_features_top{str(self.base_top)}.csv'
        self.comp_top_name = f'{self.flow_type}_{self.pred_obj}_candidate_features_top{str(self.comp_top)}.csv'
        # Convert comp_col to list if it's a string
        if isinstance(self.comp_col, str):
            self.comp_col = [self.comp_col]
        
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

    def _process_data(self):
        df = pd.read_csv(self.data_path + self.file_name)
        df = df.drop(columns = ['id', 'time'])
        pred_output = df[self.pred_obj]
        pred_input = df.drop(columns = ['step_x', 'step_y'])
        x_train, x_test, y_train, y_test = train_test_split(pred_input, pred_output, random_state = 42, test_size = 0.2, shuffle = True)
        return x_train, x_test, y_train, y_test
    
    def _load_lgb_params(self):
        param_path = self.model_path + self.param_name
        params = joblib.load(param_path)
        if not isinstance(params, dict):
            raise ValueError("Loaded parameters are not in dictionary format.") 
        return params
    
    def _train_eval_lgb(self, x_train, y_train, x_test, y_test, params, feature_list, early_stopping_rounds = 100, verbose_eval = 0):
        dtrain = lgb.Dataset(x_train[feature_list], label = y_train, free_raw_data = False)
        dvalid = lgb.Dataset(x_test[feature_list], label = y_test, reference = dtrain, free_raw_data = False)
        params = params.copy()
        params["verbosity"] = -1
        model = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds = early_stopping_rounds, verbose = bool(verbose_eval)),
                lgb.log_evaluation(period=verbose_eval) if verbose_eval else lgb.log_evaluation(period=0),],)
        preds = model.predict(x_test[feature_list], num_iteration = model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_iteration": model.best_iteration}
        return metrics

    def _incremental_feature_sets(self, base: List[str], to_add_in_order: List[str], max_add: Optional[int]):
        if max_add is None:
            max_add = len(to_add_in_order)
        for k in range(0, max_add + 1):
            yield k, base + to_add_in_order[:k]

    def _ordered_features_from_table(self):
        try:
            df_base_top = pd.read_csv(self.feat_path + self.base_top_name)
        except:
            raise ValueError('f{self.base_top_name} does not exist, please check again!')
        try:
            df_comp_top = pd.read_csv(self.feat_path + self.comp_top_name)
        except:
            raise ValueError('f{self.comp_top_name} does not exist, please check again!')
        start_base_feat_lst = df_base_top[self.base_col].tolist()
        end_base_feat_lst = df_comp_top[self.base_col].tolist()
        add_base_feat = [f for f in end_base_feat_lst if f not in start_base_feat_lst]
        
        # Handle multiple comp_cols
        add_comp_feats = {}
        for col in self.comp_col:
            if col not in df_comp_top.columns:
                raise ValueError(f"Column '{col}' not found in {self.comp_top_name}")
            end_comp_feat_lst = df_comp_top[col].tolist()
            add_comp_feats[col] = [f for f in end_comp_feat_lst if f not in start_base_feat_lst]
        
        return start_base_feat_lst, add_base_feat, add_comp_feats
    
    def _incremental_eval(self, x_train, x_test, y_train, y_test, params, base_feat_lst, add_feat_lst, max_add):
        rows = []
        base_metrics = self._train_eval_lgb(x_train, y_train, x_test, y_test, params, base_feat_lst)
        base_rmse = base_metrics['rmse']
        for k_added, feat_lst in self._incremental_feature_sets(base_feat_lst, add_feat_lst, max_add):
            m = self._train_eval_lgb(x_train, y_train, x_test, y_test, params, feat_lst)
            rows.append({
                "k_added": k_added,
                "n_features": len(feat_lst),
                "added_features": add_feat_lst[:k_added],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "delta_rmse_vs_base": m["rmse"] - base_rmse,})
        return pd.DataFrame(rows)
    
    def _performance_eval(self, max_add):
        params = self._load_lgb_params()
        x_train, x_test, y_train, y_test = self._process_data()
        start_base_feat_lst, add_base_feat, add_comp_feats = self._ordered_features_from_table()
        df_base = self._incremental_eval(x_train, x_test, y_train, y_test, params, start_base_feat_lst, add_base_feat, max_add)
        
        # Evaluate all comparison methods
        df_comps = {}
        for col, add_feat in add_comp_feats.items():
            df_comps[col] = self._incremental_eval(x_train, x_test, y_train, y_test, params, start_base_feat_lst, add_feat, max_add)
        
        return df_base, df_comps
    
    def plot_incremental_comparison(self, df_base: pd.DataFrame, df_comps: Dict[str, pd.DataFrame], metric: str = "rmse", 
                                    label_base: Optional[str] = None, label_comps: Optional[Dict[str, str]] = None, x: str = "n_features"):
        if label_base is None:
            # Use normalized name for base_col
            label_base = self._normalize_method_name(self.base_col) + " add"
        if label_comps is None:
            # Use normalized names for comparison columns
            label_comps = {col: self._normalize_method_name(col) + " add" for col in df_comps.keys()}
        
        # Validate metric and x columns
        if metric not in df_base.columns:
            raise ValueError(f"metric='{metric}' not found in df_base.")
        if x not in df_base.columns:
            raise ValueError(f"x='{x}' not found in df_base.")
        for col, df_comp in df_comps.items():
            if metric not in df_comp.columns or x not in df_comp.columns:
                raise ValueError(f"metric='{metric}' or x='{x}' not found in df_comps['{col}'].")
        
        # Sort dataframes
        dfb = df_base.sort_values(x).reset_index(drop=True)
        dfc_dict = {col: df_comp.sort_values(x).reset_index(drop=True) for col, df_comp in df_comps.items()}
        
        # Calculate x-axis range
        all_x_values = [dfb[x].min(), dfb[x].max()] + [df[x].min() for df in dfc_dict.values()] + [df[x].max() for df in dfc_dict.values()]
        x_min, x_max = int(min(all_x_values)), int(max(all_x_values))
        
        # Get color for base_col
        base_color = self._get_method_color(self.base_col)
        
        plt.figure(figsize=(10, 6))
        plt.plot(dfb[x], dfb[metric], marker="o", label=label_base, linewidth=2, color=base_color)
        
        for col, dfc in dfc_dict.items():
            label = label_comps.get(col, self._normalize_method_name(col) + " add")
            color = self._get_method_color(col)
            plt.plot(dfc[x], dfc[metric], marker="o", label=label, linewidth=2, color=color)
        
        title = f"{self.flow_type} | {self.pred_obj} | {metric} vs {x}"
        plt.xticks(range(x_min, x_max + 1))
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_name = f"{self.flow_type}_{self.pred_obj}_{metric}_{self.base_top}_{self.comp_top}_Comparison.png"
        save_path = os.path.join(self.feat_path, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)
        # force x-axis major ticks every 1
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.show()

#%%% run code
if __name__ == '__main__':
    pc = PerformanceComparer(
                            data_path = '../data/',
                            model_path = '../output_full/',
                            feat_path ='../output_single_featimp/',
                            flow_type = "corner",
                            pred_obj = "step_y",
                            base_top = 5,
                            comp_top = 25,
                            base_col = "top_shap",
                            comp_col = ["llm_selection_wo_val", "fused_importance"])  # Can now pass multiple methods
    df_base, df_comps = pc._performance_eval(max_add=20)
    pc.plot_incremental_comparison(df_base, df_comps, metric="rmse")
