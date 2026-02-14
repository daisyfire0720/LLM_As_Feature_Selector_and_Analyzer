#%%% import libraries
import os
import traceback
import accelerate
import torch
print('CUDA available:', torch.cuda.is_available()); 
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A') 
print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')
import json
import re
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import warnings
warnings.filterwarnings('ignore')

#%%% llm model configuration
class LLMConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"  
    device: str = "cuda"  
    torch_dtype: str = "float16"  # Changed to float16 to avoid FP8 quantization issues  
    max_new_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05

#%%% shap explainer with LLMs
class FeatSelector:
    def __init__(self, input_path, feat_path, output_path, flow_type, pred_obj, top_n, top_k, feat_dict):
        self.input_path = input_path
        self.feat_path = feat_path
        self.output_path = output_path
        self.flow_type = flow_type
        self.pred_obj = pred_obj
        self.llm_cfg = LLMConfig()
        self._llm = None
        self.top_n = top_n
        self.top_k = top_k
        self.feat_dict = feat_dict

    def _load_scenario_description(self):
        file_name = self.flow_type + '_scenario.txt'
        file_path = os.path.join(self.feat_path, file_name)
        with open(file_path, "r", encoding = "utf-8") as f:
            text = f.read().strip()
        if len(text) > 2000:
            text = text[:2000].rstrip() + "\n...(truncated)"
        return text

    def _process_data(self):
        self.gbm_imp =  self.flow_type + '_' + self.pred_obj + '_lightgbm_feature_importance.csv'
        self.perm_imp = self.flow_type + '_' + self.pred_obj + '_permutation_importance.csv'
        self.shap_glabal_imp = self.flow_type + '_' + self.pred_obj + '_shap_global_importance.csv'
        df_gbm = pd.read_csv(self.input_path + self.gbm_imp)
        df_perm = pd.read_csv(self.input_path + self.perm_imp)
        df_shap = pd.read_csv(self.input_path + self.shap_glabal_imp)
        df = df_gbm.merge(df_perm, on = 'feature_names').merge(df_shap, on = 'feature_names')
        df['abs_shap_imp'] = df['abs_shap_imp'] / df['abs_shap_imp'].sum()
        df['gain_feat_imp'] = df['gain_feat_imp'] / df['gain_feat_imp'].sum()
        df['split_feat_imp'] = df['split_feat_imp'] / df['split_feat_imp'].sum()
        df['perm_feat_imp'] = df['perm_feat_imp'] / df['perm_feat_imp'].sum()
        df.to_csv(self.output_path + self.flow_type + '_' + self.pred_obj + '_merged_feature_importance.csv', index = False)
        return df
    
    def _llm_discrepancy(self, df):
        df = df.sort_values(by = 'abs_shap_imp', ascending = False).head(self.top_n)
        df = df[['feature_names', 'abs_shap_imp', 'gain_feat_imp']]
        df["d_shap_minus_gain"] = df['abs_shap_imp'] - df['gain_feat_imp']
        df["d_gain_minus_shap"] = df['gain_feat_imp'] - df['abs_shap_imp']
        shap_col, gain_col = 'abs_shap_imp', 'gain_feat_imp'
        top_shap = set(df.nlargest(self.top_k, shap_col)["feature_names"])
        top_gain = set(df.nlargest(self.top_k, gain_col)["feature_names"])
        top_smg  = set(df.nlargest(self.top_k, "d_shap_minus_gain")["feature_names"])
        top_gms  = set(df.nlargest(self.top_k, "d_gain_minus_shap")["feature_names"])
        cand = df[df["feature_names"].isin(top_shap | top_gain | top_smg | top_gms)].copy()
        cand["in_top_shap"] = cand["feature_names"].isin(top_shap).astype(int)
        cand["in_top_gain"] = cand["feature_names"].isin(top_gain).astype(int)
        cand["in_top_shap_minus_gain"] = cand["feature_names"].isin(top_smg).astype(int)
        cand["in_top_gain_minus_shap"] = cand["feature_names"].isin(top_gms).astype(int)
        file_path = os.path.join(self.feat_path, 'variable_dictionary.csv')
        dd = pd.read_csv(file_path)
        dd_use = dd[dd['Variable Name'].isin(cand['feature_names'])][['Variable Name', 'Variable Meaning']]
        dd_use.rename(columns = {'Variable Name': 'feature_names', 'Variable Meaning': 'feature_meaning'}, inplace = True)
        dd_use["feature_names"] = dd_use["feature_names"].str.strip()
        dd_use['feature_meaning'] = dd_use["feature_meaning"].str.slice(0, 180)
        df = cand.merge(dd_use, on = 'feature_names', how = 'left')
        cols = ["feature_names", "gain_feat_imp", "abs_shap_imp", "d_shap_minus_gain", "d_gain_minus_shap", "in_top_shap", "in_top_gain", "in_top_shap_minus_gain", "in_top_gain_minus_shap", "feature_meaning"]
        return df[cols]
    
    def _format_llm_table(self, cand_df):
        df_use = cand_df.copy()
        for c in df_use.columns:
            if c != "feature_names" and c != "feature_meaning":
                df_use[c] = pd.to_numeric(df_use[c], errors="coerce").round(6)
        if "feature_meaning" in df_use.columns:
            df_use["feature_meaning"] = df_use["feature_meaning"].astype(str).str.slice(0, 120)
        return df_use.to_string(index=False)
    
    def _load_local_llm(self):
        if self._llm is not None:
            return 
        cfg = self.llm_cfg
        print(f"Loading LLM model: {cfg.model_name}")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast = True)
            self._model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                device_map = "auto" if cfg.device == "cuda" else None,
                torch_dtype = cfg.torch_dtype,)
            self._llm = pipeline(
                        task = "text-generation",
                        model = self._model,
                        tokenizer=self._tokenizer,)
            print("LLM loaded successfully")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            raise
    
    def _build_prompt_feature_selector(self, scen_context: str, feature_table: str) -> str:
        prompt_file = os.path.join(os.path.dirname(__file__), 'featselect_prompt.txt')
        final_n, n_core, n_rare, n_art = self.feat_dict.get('final_n'), self.feat_dict.get('n_core'), self.feat_dict.get('n_rare'), self.feat_dict.get('n_art')
        with open(prompt_file, 'r', encoding = 'utf-8') as f:
            prompt_template = f.read()
        prompt_body = prompt_template.format(pred_obj = self.pred_obj, 
                                             scenario_name = self.flow_type, 
                                             scenario_context = scen_context,
                                             k = self.top_k,
                                             final_n = final_n,
                                             n_core = n_core,
                                             n_rare = n_rare,
                                             n_art = n_art,
                                             feature_table = feature_table).strip()
        prompt = (
                    "<s>[INST]\n"
                    + prompt_body
                    + "\n\nIMPORTANT:\n"
                    "Your response MUST start with '{{' and MUST end with '}}'.\n"
                    "[/INST]"
                )
        return prompt
    
    def _extract_json(self, text: str) -> dict:
        raw = text.strip()

        def _try_load(s: str):
            try:
                obj = json.loads(s)
            except Exception:
                return None
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return obj
            return obj

        parsed = _try_load(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            raw = parsed.strip()
            parsed = _try_load(raw)
            if isinstance(parsed, dict):
                return parsed

        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            parsed = _try_load(candidate)
            if isinstance(parsed, dict):
                return parsed

        start = raw.find("{")
        if start == -1:
            raise ValueError("No JSON object found: missing '{'.\nFirst 400 chars:\n" + raw[:400])

        brace_count = 0
        end = None
        for i in range(start, len(raw)):
            if raw[i] == "{":
                brace_count += 1
            elif raw[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if end is None:
            raise ValueError(
                "JSON appears to be TRUNCATED (no matching closing '}').\n"
                "Likely causes:\n"
                "- max_new_tokens too small\n"
                "- schema/output too large\n\n"
                "First 400 chars:\n" + raw[:400]
            )

        candidate = raw[start:end + 1]
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]
        candidate = candidate.replace('\"', '"')

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            return parsed
        except Exception as e:
            raise ValueError("Found JSON-like block but parsing failed.\nFirst 400 chars:\n" + raw[:400]) from e

    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        self._load_local_llm()
        cfg = self.llm_cfg
        out = self._llm(
            prompt,
            max_new_tokens = cfg.max_new_tokens,
            temperature = cfg.temperature,
            top_p = cfg.top_p,
            repetition_penalty = cfg.repetition_penalty,
            do_sample = False,
            return_full_text = False,)
        text = out[0]["generated_text"]
        return self._extract_json(text)

    def llm_feature_selector(self, df_merged: pd.DataFrame):
        scen_context = self._load_scenario_description()
        df_use = self._llm_discrepancy(df_merged)
        feature_table = self._format_llm_table(df_use)
        prompt = self._build_prompt_feature_selector(scen_context, feature_table)
        result = self._generate_json(prompt)
        return result
    
    def all_feature_selector(self, save = False):
        df_merged = self._process_data()
        candidate_dict = {}
        # baseline 1: top gain features
        gain_cand = df_merged.sort_values(by = 'gain_feat_imp', ascending = False).head(self.feat_dict['final_n'])['feature_names'].tolist()
        candidate_dict['top_gain'] = gain_cand
        # baseline 2: top shap features
        shap_cand = df_merged.sort_values(by = 'abs_shap_imp', ascending = False).head(self.feat_dict['final_n'])['feature_names'].tolist()
        candidate_dict['top_shap'] = shap_cand
        # llm based feature selection
        llm_result = self.llm_feature_selector(df_merged)
        candidate_dict['llm_selection'] = llm_result.get('selected_features', [])
        # weighted fusion metric
        shap_weight, gain_weight, split_weight, perm_weight = 0.4, 0.3, 0.1, 0.2 
        df_merged['fused_imp'] = (shap_weight * df_merged['abs_shap_imp'] + 
                                  gain_weight * df_merged['gain_feat_imp'] + 
                                  split_weight * df_merged['split_feat_imp'] + 
                                  perm_weight * df_merged['perm_feat_imp'])
        fused_cand = df_merged.sort_values(by = 'fused_imp', ascending = False).head(self.feat_dict['final_n'])['feature_names'].tolist()
        candidate_dict['fused_importance'] = fused_cand
        if save:
            df_cand = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in candidate_dict.items()]))
            df_cand.to_csv(self.output_path + f'{self.flow_type}_{self.pred_obj}_candidate_features_top{self.feat_dict["final_n"]}.csv', index = False)
        return candidate_dict

    def all_feature_analyzer(self, candidate_dict):
        df_cand = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in candidate_dict.items()]))
        methods = df_cand.columns.tolist()
        feat_sets = {m: set(df_cand[m].dropna()) for m in methods}
        shared_all = set.intersection(*feat_sets.values())
        unique_by_method = {m: feat_sets[m] - set.union(*(feat_sets[k] for k in methods if k != m)) for m in methods}
        overlap = pd.DataFrame(index = methods, columns = methods, dtype=int)
        for m1 in methods:
            for m2 in methods:
                overlap.loc[m1, m2] = len(feat_sets[m1] & feat_sets[m2])
        def jaccard(a, b):
            return len(a & b) / len(a | b)
        jaccard_scores = pd.DataFrame(index = methods, columns = methods, dtype = float)
        for m1 in methods:
            for m2 in methods:
                jaccard_scores.loc[m1, m2] = jaccard(feat_sets[m1], feat_sets[m2])
        return shared_all, unique_by_method, overlap, jaccard_scores    

#%%% main function for testing
if __name__ == '__main__':
    input_path = '../output_full/'
    feat_path = ''
    output_path = '../output_feature_importance/'
    flow_type = 'corner'
    pred_obj = 'step_x'
    top_n = 50
    top_k = 25
    feat_dict = {'final_n': 10, 'n_core': 5, 'n_rare': 3, 'n_art': 2}
    feat_dict = {'final_n': 20, 'n_core': 10, 'n_rare': 6, 'n_art': 4}
    #feat_dict = {'final_n': 25, 'n_core': 12, 'n_rare': 8, 'n_art': 5}
    try:
        # Your code here
        explainer = FeatSelector(input_path, feat_path, output_path, flow_type, pred_obj, top_n, top_k, feat_dict)
        candidate_dict = explainer.all_feature_selector(save = True)
        print("Candidate feature sets:")
    except KeyError as e:
        traceback.print_exc()


# %%
