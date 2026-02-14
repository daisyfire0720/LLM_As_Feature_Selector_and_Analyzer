#%%% import libraries
import os
import traceback
import json
import re
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

#%% define class
class FlowValidator:
    def __init__(self, data_path, model_path, feat_path, flow_type, pred_obj):
        self.data_path = data_path
        self.model_path = model_path
        self.feat_path = feat_path
        self.flow_type = flow_type
        self.pred_obj = pred_obj
        self.file_name = 'ds_vector_' + flow_type + '.csv'

    def _process_data(self):
        df = pd.read_csv(self.data_path + self.file_name)
        df = df.drop(columns = ['id', 'time'])
        pred_output = df[self.pred_obj]
        pred_input = df.drop(columns = ['step_x', 'step_y'])
        input_train, input_test, output_train, output_test = train_test_split(pred_input, pred_output, random_state = 42, test_size = 0.2, shuffle = True)
        return input_train, output_train, input_test, output_test
    
    def _load_lgb_params(self):
        param_path = self.model_path + f'{self.flow_type}_{self.pred_obj}_lgb_param.joblib'
        params = joblib.load(param_path)
        if not isinstance(params, dict):
            raise ValueError("Loaded parameters are not in dictionary format.") 
        return params
    
    def _load_candidate_features(self, method_col, top_k):
        feat_path = self.feat_path + f'{self.flow_type}_{self.pred_obj}_candidate_features_top{str(top_k)}.csv'
        df_feat = pd.read_csv(feat_path)
        if method_col not in df_feat.columns:
            raise ValueError(f"method_col='{method_col}' not in {df_feat.columns.tolist()}")
        feat_lst = df_feat[method_col].dropna().astype(str).tolist()
        # Load the data to verify which features actually exist
        df = pd.read_csv(self.data_path + self.file_name)
        # Filter to only keep features that exist in the data
        valid_feats = [f for f in feat_lst if f in df.columns]
        if len(valid_feats) < len(feat_lst):
            invalid = [f for f in feat_lst if f not in df.columns]
            print(f"Warning: Filtered out invalid features: {invalid}")
        return valid_feats
    
    def _train_eval_default_lgb(self, x_train, y_train, x_test, y_test, feature_list, early_stopping_rounds = 100, verbose_eval = 0):
        dtrain = lgb.Dataset(x_train[feature_list], label = y_train, free_raw_data = False)
        dvalid = lgb.Dataset(x_test[feature_list], label = y_test, reference = dtrain, free_raw_data = False)
        default_model = lgb.train(
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "min_data_in_leaf": 20,
                "verbosity": -1
            },
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds = early_stopping_rounds, verbose = bool(verbose_eval)),
                lgb.log_evaluation(period=verbose_eval) if verbose_eval else lgb.log_evaluation(period=0),],)
        preds = default_model.predict(x_test[feature_list], num_iteration = default_model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_iteration": default_model.best_iteration}
        return default_model, metrics

    def _train_eval_lgb(self, x_train, y_train, x_test, y_test, params, feature_list, early_stopping_rounds = 100, verbose_eval = 0):
        dtrain = lgb.Dataset(x_train[feature_list], label = y_train, free_raw_data = False)
        dvalid = lgb.Dataset(x_test[feature_list], label = y_test, reference = dtrain, free_raw_data = False)
        params = params.copy()
        params['verbosity'] = -1
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
        return model, metrics
    
    def compare_feature_sets(self, method_cols: list, top_k: int, plot: bool = False, early_stopping_rounds: int = 100) -> pd.DataFrame:
        X_train, y_train, X_test, y_test = self._process_data()
        params = self._load_lgb_params()
        results = []
        # selected-feature baselines
        for col in method_cols:
            feats = self._load_candidate_features(method_col = col, top_k=top_k)
            _, m = self._train_eval_lgb(
                X_train, y_train, X_test, y_test,
                params=params,
                feature_list=feats,
                early_stopping_rounds=early_stopping_rounds)
            results.append({"method": col, **m})

        res_df = pd.DataFrame(results).sort_values("rmse")
        # optional plot (no seaborn requirement—matplotlib only)
        if plot:
            plt.figure()
            plt.bar(res_df["method"], res_df["rmse"])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("RMSE")
            plt.title(f"{self.flow_type} | {self.pred_obj} | RMSE by feature selection")
            plt.tight_layout()
            plt.show()
        # save
        os.makedirs(self.feat_path, exist_ok=True)
        out_csv = os.path.join(self.feat_path, f"{self.flow_type}_{self.pred_obj}_{self.top_k}_feature_selection_comparison.csv")
        res_df.to_csv(out_csv, index=False)
        return res_df
    
    def compare_feature_sets_over_topn(self, method_cols: list, top_n_cols: list, metric: str = "rmse", early_stopping_rounds: int = 100, plot: bool = True,) -> pd.DataFrame:
        X_train, y_train, X_test, y_test = self._process_data()
        params = self._load_lgb_params()
        rows = []
        # loop candidate files
        for col in method_cols:
            for k in top_n_cols:
                feats = self._load_candidate_features(method_col=col, top_k=k)
                # Some columns may have fewer than top_n due to NaN padding; that's OK
                _, m = self._train_eval_lgb(
                    X_train, y_train, X_test, y_test,
                    params=params,
                    feature_list=feats,
                    early_stopping_rounds=early_stopping_rounds)
                rows.append({
                    "method": col,
                    "top_n": k,
                    **m})
        res_long = pd.DataFrame(rows)
        # Save
        os.makedirs(self.feat_path, exist_ok=True)
        out_csv = os.path.join(
            self.feat_path,
            f"{self.flow_type}_{self.pred_obj}_rmse_vs_topn.csv")
        res_long.to_csv(out_csv, index=False)

        # Plot 4 polylines (one per method)
        if plot:
            plt.figure()
            # only plot the feature-selection methods (exclude all_features baseline)
            plot_df = res_long[res_long["method"].isin(method_cols)].copy()
            plot_df = plot_df.sort_values(["method", "top_n"])
            for m in method_cols:
                sub = plot_df[plot_df["method"] == m]
                plt.plot(sub["top_n"], sub[metric], marker="o", label=m)
            # Add a horizontal baseline line for all-features
            plt.xlabel("Top-N features")
            plt.ylabel(metric.upper())
            plt.title(f"{self.flow_type} | {self.pred_obj} | {metric.upper()} vs Top-N")
            plt.legend()
            plt.tight_layout()
            out_png = os.path.join(self.feat_path, f"{self.flow_type}_{self.pred_obj}_{metric}_vs_topn.png")
            plt.savefig(out_png)
            plt.show()
        return res_long


#%%% main module
if __name__ == '__main__':
    data_path = '../data/'
    model_path = '../output_full/'
    feat_path = '../output_single_featimp/'
    flow_type_lst = ['intersection']
    pred_obj_lst = ['step_y']
    method_cols = ['top_shap','llm_selection', 'llm_selection_wo', 'llm_selection_wo_val', 'llm_selection_wo_both','fused_importance']
    top_n_cols = [5, 10, 15, 20, 25]
    for flow_type in flow_type_lst:
        for pred_obj in pred_obj_lst:
            model = FlowValidator(data_path, model_path, feat_path, flow_type, pred_obj)
            compare_feature_sets_over_topn = model.compare_feature_sets_over_topn(method_cols = method_cols, top_n_cols = top_n_cols)


# %%
