
#%%% import packages 
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

#%%% define data class for use
@dataclass
class SelectionMechanismAnalyzer:
    data_path: str
    model_path: str
    feat_path: str
    flow_type: str
    pred_obj: str
    base_col: str
    comp_col: str
    
    def __post_init__(self):
        self.file_name = f"ds_vector_{self.flow_type}.csv"
        self.param_name = f"{self.flow_type}_{self.pred_obj}_lgb_param.joblib"
        self.feat_file = f"{self.flow_type}_{self.pred_obj}_candidate_features_top25.csv"
    
    def _load_data(self):
        df = pd.read_csv(self.data_path + self.file_name)
        df = df.drop(columns=['id','time'])
        y = df[self.pred_obj]
        X = df.drop(columns=['step_x','step_y'])
        return X, y
    
    def _load_params(self):
        return joblib.load(self.model_path + self.param_name)
    
    def effective_snr_by_method(self, feature_list, n_splits=5):
        X, y = self._load_data()
        params = self._load_params()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train[feature_list], y_train)
            r2_scores.append(r2_score(y_test, model.predict(X_test[feature_list])))
        
        mean_r2 = np.mean(r2_scores)
        snr = mean_r2 / (1 - mean_r2)
        
        return mean_r2, snr
    
    def ranking_agreement(self):
        df = pd.read_csv(self.feat_path + self.feat_file)
        shap_rank = df[self.base_col].tolist()
        llm_rank = df[self.comp_col].tolist()
        
        # Convert to ranking index
        shap_idx = {f:i for i,f in enumerate(shap_rank)}
        llm_idx = [shap_idx[f] for f in llm_rank if f in shap_idx]
        
        tau, _ = kendalltau(range(len(llm_idx)), llm_idx)
        
        return tau