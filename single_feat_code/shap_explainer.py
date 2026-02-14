#%%% import libraries
import os
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
    max_new_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05

#%%% shap explainer with LLMs
class FeatDiffExplainer:
    def __init__(self, input_path, feat_path, output_path, flow_type, pred_obj):
        self.input_path = input_path
        self.feat_path = feat_path
        self.output_path = output_path
        self.flow_type = flow_type
        self.pred_obj = pred_obj
        self.llm_cfg = LLMConfig()
        self._llm = None
    
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
    
    def _load_scenario_description(self):
        file_name = self.flow_type + '_scenario.txt'
        file_path = os.path.join(self.feat_path, file_name)
        with open(file_path, "r", encoding = "utf-8") as f:
            text = f.read().strip()
        if len(text) > 2000:
            text = text[:2000].rstrip() + "\n...(truncated)"
        return text
    
    def _load_schema(self):
        file_path = os.path.join(self.feat_path, 'featdiff_schema.txt')
        with open(file_path, "r", encoding = 'utf-8') as f:
            text = f.read().strip()
        return text

    def _format_data_dictionary_table(self, df, top_k: int = 25, sort_col: str = "abs_shap_imp", include_cols: Optional[List[str]] = None):
        include_cols = include_cols or ["feature_names", "split_feat_imp", "gain_feat_imp", "perm_feat_imp", "abs_shap_imp", "shap_imp"]
        df_use = df[include_cols].copy()
        if sort_col in df_use.columns:
            df_use = df_use.sort_values(sort_col, ascending = False)
        df_use = df_use.head(top_k)
        top_features = df_use['feature_names'].tolist()
        file_path = os.path.join(self.feat_path, 'variable_dictionary.csv')
        dd = pd.read_csv(file_path)
        dd_use = dd[dd['Variable Name'].isin(top_features)][['Variable Name', 'Variable Meaning']]
        order = {f: i for i, f in enumerate(top_features)}
        dd_use["_order"] = dd_use["Variable Name"].map(order)
        dd_use = dd_use.sort_values("_order").drop(columns="_order")
        dd_use["Variable Name"] = dd_use["Variable Name"].str.strip()
        dd_use['Variable Meaning'] = dd_use["Variable Meaning"].str.slice(0, 180)
        return dd_use.to_string(index=False)

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
        
    def _format_feature_table(self, df, top_k: int = 25, sort_col: str = "abs_shap_imp", include_cols: Optional[List[str]] = None):
        include_cols = include_cols or ["feature_names", "split_feat_imp", "gain_feat_imp", "perm_feat_imp", "abs_shap_imp", "shap_imp"]
        df_use = df[include_cols].copy()
        if sort_col in df_use.columns:
            df_use = df_use.sort_values(sort_col, ascending=False)
        df_use = df_use.head(top_k)
        for col in df_use.columns:
            if col != "feature_names":
                df_use[col] = pd.to_numeric(df_use[col], errors="coerce").round(6)
        return df_use.to_string(index = False)

    def _build_prompt_global_discrepancy(self, scen_context: str, data_dictionary: str, schema: str, feature_table: str) -> str:
        prompt_file = os.path.join(os.path.dirname(__file__), 'featdiff_prompt.txt')
        with open(prompt_file, 'r', encoding = 'utf-8') as f:
            prompt_template = f.read()
        prompt_body = prompt_template.format(pred_obj = self.pred_obj, 
                                             scenario_name = self.flow_type, 
                                             scenario_context = scen_context,
                                             data_dictionary = data_dictionary,
                                             schema = schema, 
                                             feature_table = feature_table).strip()
        prompt = (
                    "<s>[INST]\n"
                    + prompt_body
                    + "\n\nIMPORTANT:\n"
                    "Your response MUST start with '{' and MUST end with '}'.\n"
                    "Do NOT include any text outside the JSON object.\n"
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
        print("===PROMPT BEGIN===")
        print(prompt[:1000])  # print first 1000 chars of prompt
        print("===PROMPT END===")
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
        print("===RAW LLM OUTPUT BEGIN===")
        print(text[:2000])
        print("===RAW LLM OUTPUT END===")
        return self._extract_json(text)

    def explain_global_discrepancy(self, df_merged: pd.DataFrame, top_k: int = 25, sort_col: str = "abs_shap_imp", save_json: bool = True, out_name: str = "global_discrepancy_explanations.json") -> Dict[str, Any]:
        """
        Explain discrepancies between SHAP and built-in importance metrics at global level.
        """
        scen_context = self._load_scenario_description()
        data_dictionary = self._format_data_dictionary_table(df_merged, top_k = top_k, sort_col = sort_col)
        schema = self._load_schema()
        feature_table = self._format_feature_table(df_merged, top_k = top_k, sort_col = sort_col)
        prompt = self._build_prompt_global_discrepancy(scen_context, data_dictionary, schema, feature_table)
        result = self._generate_json(prompt)

        if save_json:
            os.makedirs(self.output_path, exist_ok = True)
            out_path = os.path.join(self.output_path, out_name)
            with open(out_path, "w", encoding ="utf-8") as f:
                json.dump(result, f, indent=2)
        return result
    
    def main(self):
        df_merged = self._process_data()
        result = self.explain_global_discrepancy(df_merged, top_k = 25, sort_col = "abs_shap_imp", save_json = True, out_name = f"{self.flow_type}_{self.pred_obj}_global_discrepancy_explanations.json")
        return result
    
# %% run module
if __name__ == '__main__':
    input_path = '../output_full/'
    feat_path = ''
    output_path = '../output_feature_importance/'
    flow_type = 'corner'
    pred_obj = 'step_x'
    explainer = FeatDiffExplainer(input_path, feat_path, output_path, flow_type, pred_obj)
    result = explainer.main()
    print(result)


# %%
# %% run module
if __name__ == '__main__':
    input_path = '../output_full/'
    feat_path = ''
    output_path = '../output_feature_importance/'
    flow_type = 'corner'
    pred_obj = 'step_x'
    explainer = FeatDiffExplainer(input_path, feat_path, output_path, flow_type, pred_obj)
    explainer._process_data()
# %%
