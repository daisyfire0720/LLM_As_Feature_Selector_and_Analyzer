#%%% import libraries
import os
import time
import traceback
import json
import re
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import warnings
warnings.filterwarnings('ignore')

#%%% llm model configuration
class LLMConfig:
    # Auth / model
    api_key_env: str = "GEMINI_API_KEY"
    api_key: str = os.getenv("GEMINI_API_KEY", "AIzaSyBXNR0G23BDwX9Nfq5WP_1FBspyE5XwHT8") 
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") 

    # Generation params
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "20000"))
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("GEMINI_TOP_P", "0.9"))
    # MIME type for structured JSON output
    response_mime_type: str = "application/json"
    # Networking
    timeout_s: int = int(os.getenv("GEMINI_TIMEOUT_S", "60"))

#%%% shap explainer with LLMs
class FeatSelector:
    def __init__(self, input_path, feat_path, output_path, flow_type, pred_obj, top_n, top_k, final_n):
        self.input_path = input_path
        self.feat_path = feat_path
        self.output_path = output_path
        self.flow_type = flow_type
        self.pred_obj = pred_obj
        self.llm_cfg = LLMConfig()
        self._llm = None
        self.top_n = top_n
        self.top_k = top_k
        self.final_n = final_n

    def _load_scenario_description(self):
        file_name = self.flow_type + '_scenario.txt'
        file_path = os.path.join(self.feat_path, file_name)
        with open(file_path, "r", encoding = "utf-8") as f:
            text = f.read().strip()
        if len(text) > 2000:
            text = text[:2000].rstrip() + "\n...(truncated)"
        return text

    def _process_data(self):
        self.gbm_imp = 'merged_' + self.pred_obj + '_lightgbm_feature_importance.csv'
        self.perm_imp = 'merged_' + self.pred_obj + '_permutation_importance.csv'
        self.shap_glabal_imp = 'merged_' + self.pred_obj + '_shap_global_importance.csv'
        df_gbm = pd.read_csv(self.input_path + self.gbm_imp)
        df_perm = pd.read_csv(self.input_path + self.perm_imp)
        df_shap = pd.read_csv(self.input_path + self.shap_glabal_imp)
        df = df_gbm.merge(df_perm, on = 'feature_names').merge(df_shap, on = 'feature_names')
        df['abs_shap_imp'] = df['abs_shap_imp'] / df['abs_shap_imp'].sum()
        df['gain_feat_imp'] = df['gain_feat_imp'] / df['gain_feat_imp'].sum()
        df['split_feat_imp'] = df['split_feat_imp'] / df['split_feat_imp'].sum()
        df['perm_feat_imp'] = df['perm_feat_imp'] / df['perm_feat_imp'].sum()
        df.to_csv(self.output_path + 'merged_' + self.pred_obj + '_merged_feature_importance.csv', index = False)
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
        return df_use.to_string(index = False)
    
    def _build_prompt_feature_selector(self, scen_context: str, feature_table: str, with_domain: bool, with_value: bool) -> str:
        if with_domain is True and with_value is True:
            prompt_file = os.path.join(self.feat_path, 'merge_feat_prompt', 'merge_featselect.txt')
        if with_domain is False and with_value is True:
            prompt_file = os.path.join(self.feat_path, 'merge_feat_prompt', 'merge_featselect_wo_scen.txt')
        if with_domain is True and with_value is False:
            prompt_file = os.path.join(self.feat_path, 'merge_feat_prompt', 'merge_featselect_wo_val.txt')
        if with_domain is False and with_value is False:
            prompt_file = os.path.join(self.feat_path, 'merge_feat_prompt', 'merge_featselect_wo_scen_val.txt')
        with open(prompt_file, 'r', encoding = 'utf-8') as f:
            prompt_template = f.read()
        prompt_body = prompt_template.format(pred_obj = self.pred_obj, 
                                             scenario_name = self.flow_type,
                                             scenario_context = scen_context,
                                             k = self.top_k,
                                             final_n = self.final_n,
                                             feature_table = feature_table).strip()
        return prompt_body
    
    def _extract_json(self, text: str) -> dict:
        raw = text.strip()

        def _normalize(obj):
            if isinstance(obj, dict):
                new = {}
                for k, v in obj.items():
                    nk = k
                    if isinstance(nk, str):
                        nk = nk.strip()
                        if nk.startswith('"') and nk.endswith('"'):
                            nk = nk[1:-1]
                        nk = nk.strip()
                    new[nk] = _normalize(v)
                return new
            if isinstance(obj, list):
                return [_normalize(x) for x in obj]
            return obj
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

        # 1) Fast path: direct JSON or double-encoded JSON
        parsed = _try_load(raw)
        if isinstance(parsed, dict):
            return _normalize(parsed)
        if isinstance(parsed, str):
            raw = parsed.strip()
            parsed = _try_load(raw)
            if isinstance(parsed, dict):
                return _normalize(parsed)

        # 2) Markdown ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            parsed = _try_load(candidate)
            if isinstance(parsed, dict):
                return _normalize(parsed)

        # 3) Find first { and match braces
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
        # Strip surrounding double-quotes if the block was accidentally quoted
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]
        # Unescape any escaped quotes inside
        candidate = candidate.replace('\"', '"')

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return _normalize(parsed)
            return _normalize(parsed) if isinstance(parsed, (dict, list)) else parsed
        except Exception as e:
            raise ValueError("Found JSON-like block but parsing failed.\nFirst 400 chars:\n" + raw[:400]) from e

    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        cfg = self.llm_cfg

        # 1. Setup API Key
        # Prefer fetching from os.environ at runtime to ensure it captures updates
        api_key = cfg.api_key or os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {cfg.api_key_env}. Please set it before running.")
        
        genai.configure(api_key=api_key)

        # 2. Configure Generation Settings
        generation_config = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_output_tokens": cfg.max_output_tokens,
            "response_mime_type": cfg.response_mime_type,
        }

        # 3. Initialize Model
        # We explicitly disable safety filters here to prevent the model from 
        # refusing to process benign data features that might look "statistical" or "medical".
        model = genai.GenerativeModel(
            model_name=cfg.model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        try:
            # 4. Generate Content
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": cfg.timeout_s}
            )
            
            # 5. Extract Text
            # response.text accesses the result. Raises an error if blocked by safety filters.
            return self._extract_json(response.text)

        except Exception as e:
            # Basic error handling
            print(f"Gemini API Error: {str(e)}")
            # If the model blocked the response, response.prompt_feedback might help debug
            raise e


    def llm_feature_selector(self, df_merged: pd.DataFrame, with_domain: bool, with_value: bool) -> Dict[str, Any]:
        scen_context = self._load_scenario_description()
        if with_value:
            df_use = self._llm_discrepancy(df_merged)
            feature_table = self._format_llm_table(df_use)
        else:
            df_use = self._llm_discrepancy(df_merged)
            df_use = df_use[['feature_names', 'feature_meaning']]
            feature_table = self._format_llm_table(df_use)
        prompt = self._build_prompt_feature_selector(scen_context, feature_table, with_domain, with_value)
        result = self._generate_json(prompt)
        return result
    
    def all_feature_selector(self, save = False):
        df_merged = self._process_data()
        candidate_dict = {}
        # baseline 1: top gain features
        gain_cand = df_merged.sort_values(by = 'gain_feat_imp', ascending = False).head(self.final_n)['feature_names'].tolist()
        candidate_dict['top_gain'] = gain_cand
        # baseline 2: top shap features
        shap_cand = df_merged.sort_values(by = 'abs_shap_imp', ascending = False).head(self.final_n)['feature_names'].tolist()
        candidate_dict['top_shap'] = shap_cand
        # baseline 3: llm based feature selection with domain context   
        llm_result = self.llm_feature_selector(df_merged, with_domain = True, with_value = True)
        candidate_dict['llm_selection'] = llm_result.get('selected_features', [])
        # baseline 4: llm based feature selection without domain context
        llm_result_wo_scen = self.llm_feature_selector(df_merged, with_domain = False, with_value = True)
        candidate_dict['llm_selection_wo'] = llm_result_wo_scen.get('selected_features', [])
        # baseline 5: llm based feature selection with domain context without feature values
        llm_result_wo_val = self.llm_feature_selector(df_merged, with_domain = True, with_value = False)
        candidate_dict['llm_selection_wo_val'] = llm_result_wo_val.get('selected_features', [])
        # baseline 6: llm based feature selection without domain context and without feature values
        llm_result_wo_scen_val = self.llm_feature_selector(df_merged, with_domain = False, with_value = False)
        candidate_dict['llm_selection_wo_both'] = llm_result_wo_scen_val.get('selected_features', [])
        # weighted fusion metric
        shap_weight, gain_weight, split_weight, perm_weight = 0.4, 0.3, 0.1, 0.2 
        df_merged['fused_imp'] = (shap_weight * df_merged['abs_shap_imp'] + 
                                  gain_weight * df_merged['gain_feat_imp'] + 
                                  split_weight * df_merged['split_feat_imp'] + 
                                  perm_weight * df_merged['perm_feat_imp'])
        fused_cand = df_merged.sort_values(by = 'fused_imp', ascending = False).head(self.final_n)['feature_names'].tolist()
        candidate_dict['fused_importance'] = fused_cand
        if save:
            df_cand = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in candidate_dict.items()]))
            # Debug: print resulting DataFrame columns
            print("df_cand.columns:", list(df_cand.columns))
            df_cand.to_csv(self.output_path + f'{self.flow_type}_{self.pred_obj}_candidate_features_top{self.final_n}.csv', index = False)
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
# if __name__ == '__main__':
#     os.environ["GEMINI_API_KEY"] = "AIzaSyBXNR0G23BDwX9Nfq5WP_1FBspyE5XwHT8"
#     os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
#     os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
#     if not os.getenv("GEMINI_API_KEY"):
#         raise RuntimeError("Please export GEMINI_API_KEY before running.")
#     input_path = '../output_sliced/'
#     feat_path = ''
#     output_path = '../output_feature_importance/'
#     top_n = 50
#     top_k = 25
#     final_n = 10
#     pred_obj = 'step_x'
#     flow_type = 'corner'
#     try:
#         # Your code here
#         explainer = FeatSelector(input_path, feat_path, output_path, flow_type, pred_obj, top_n, top_k, final_n)
#         candidate_dict = explainer.all_feature_selector(save = True)
#         print("Sucssessfully generated candidate features for", flow_type, pred_obj, final_n)
#         # rest 2 minute
#         time.sleep(60)
#     except KeyError as e:
#         traceback.print_exc()


#%%% loop for use
if __name__ == '__main__':
    os.environ["GEMINI_API_KEY"] = "AIzaSyBXNR0G23BDwX9Nfq5WP_1FBspyE5XwHT8"
    os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
    os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Please export GEMINI_API_KEY before running.")
    input_path = '../output_sliced/'
    feat_path = ''
    output_path = '../output_sliced_featimp/'
    top_n = 50
    top_k = 25
    final_n_lst = [25]
    pred_obj_lst = ['step_x', 'step_y']
    flow_type_lst = ['corner', 'entrance', 'intersection']
    for pred_obj in pred_obj_lst: 
        for flow_type in flow_type_lst:
            for final_n in final_n_lst:
                try:
                    # Your code here
                    explainer = FeatSelector(input_path, feat_path, output_path, flow_type, pred_obj, top_n, top_k, final_n)
                    candidate_dict = explainer.all_feature_selector(save = True)
                    print("Successfully generated candidate features for", flow_type, pred_obj, final_n)
                    # rest 2 minute
                    time.sleep(90)
                except KeyError as e:
                    traceback.print_exc()

# %%
