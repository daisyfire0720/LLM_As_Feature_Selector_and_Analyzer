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
    torch_dtype: str = "auto"  
    max_new_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05

#%%% scenario explainer with llm
class ScenDiffExplainer:
    def __init__(self, input_path, feat_path, output_path, flow_type_lst, pred_obj):
        self.input_path = input_path
        self.feat_path = feat_path
        self.output_path = output_path
        self.flow_type_lst = flow_type_lst
        self.pred_obj = pred_obj
        self.llm_cfg = LLMConfig()
        self._llm = None


    def _process_data(self) -> Dict[str, pd.DataFrame]:
        dfs = {}
        for flow_type in self.flow_type_lst:
            fname = f"{flow_type}_{self.pred_obj}_shap_global_importance.csv"
            fpath = os.path.join(self.input_path, fname)
            df = pd.read_csv(fpath)
            denom = df["abs_shap_imp"].sum()
            df["abs_shap_norm"] = df["abs_shap_imp"] / (denom + 1e-12)
            dfs[flow_type] = df
        return dfs

    def _load_scenario_description(self):
        dfs = {}
        for flow_type in self.flow_type_lst:
            file_name = flow_type + '_scenario.txt'
            file_path = os.path.join(self.feat_path, file_name)
            with open(file_path, "r", encoding = "utf-8") as f:
                text = f.read().strip()
            if len(text) > 2000:
                text = text[:2000].rstrip() + "\n...(truncated)"
            dfs[flow_type] = text
        return dfs

    def _load_schema(self):
        file_path = os.path.join(self.feat_path, 'scendiff_schema.txt')
        with open(file_path, "r", encoding = 'utf-8') as f:
            text = f.read().strip()
        return text

    def _load_local_llm(self):
        if self._llm is not None:
            return
        cfg = self.llm_cfg
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto" if cfg.device == "cuda" else None,
            torch_dtype=cfg.torch_dtype,
        )
        self._llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=(cfg.temperature > 0),
            return_full_text=False,
        )
        text = out[0]["generated_text"]
        return self._extract_json(text)

    def _format_feature_table(self, df: pd.DataFrame, top_k: int = 25) -> str:
        cols = ["feature_names", "abs_shap_imp", "abs_shap_norm", "shap_imp"]
        df2 = df[cols].copy()
        df2 = df2.sort_values("abs_shap_norm", ascending = False).head(top_k)
        file_path = os.path.join(self.feat_path, 'variable_dictionary.csv')
        dd = pd.read_csv(file_path)
        dd.rename(columns = {'Variable Name' : 'feature_names'}, inplace = True)
        df2 = df2.merge(dd, left_on = "feature_names", right_on = "feature_names", how = "left")
        for c in ["abs_shap_imp", "abs_shap_norm", "shap_imp"]:
            df2[c] = pd.to_numeric(df2[c], errors = "coerce").round(6)
        return df2.to_string(index=False)
    
    def _build_prompt_scenario_discrepancy(self, scen_a, scen_b, scen_context_a, scen_context_b, schema, table_a, table_b) -> str:
        prompt_file = os.path.join(os.path.dirname(__file__), 'scendiff_prompt.txt')
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        prompt_body = prompt_template.format(pred_obj = self.pred_obj,
                                             scenario_a = scen_a, 
                                             scenario_b = scen_b, 
                                             scenario_context_a = scen_context_a,
                                             scenario_context_b = scen_context_b,
                                             table_a = table_a, 
                                             table_b = table_b, 
                                             schema = schema).strip()
        prompt = (
                    "<s>[INST]\n"
                    + prompt_body
                    + "\n\nIMPORTANT:\n"
                    "Your response MUST start with '{' and MUST end with '}'.\n"
                    "Do NOT include any text outside the JSON object.\n"
                    "[/INST]"
                )
        return prompt
    
    def explain_scenario_discrepancy(self, dfs, scen_a: str,  scen_b: str, top_k_table: int = 25, save_json: bool = True, out_name: Optional[str] = None,) -> Dict[str, Any]:
        if scen_a not in dfs or scen_b not in dfs:
            raise ValueError(f"Scenario not found. Available: {list(dfs.keys())}")
        df_a = dfs[scen_a].copy()
        df_b = dfs[scen_b].copy()
        df_a = df_a.sort_values("abs_shap_norm", ascending=False).reset_index(drop=True)
        df_b = df_b.sort_values("abs_shap_norm", ascending=False).reset_index(drop=True)
        scen_context_a = self._load_scenario_description()[scen_a]
        scen_context_b = self._load_scenario_description()[scen_b]
        table_a = self._format_feature_table(df_a, top_k = top_k_table)
        table_b = self._format_feature_table(df_b, top_k = top_k_table)
        schema = self._load_schema()
        prompt = self._build_prompt_scenario_discrepancy(scen_a, scen_b, scen_context_a, scen_context_b, schema, table_a, table_b)
        result = self._generate_json(prompt)
        if save_json:
            os.makedirs(self.output_path, exist_ok = True)
            if out_name is None:
                out_name = f"scenario_diff_{self.pred_obj}_{scen_a}_vs_{scen_b}.json"
            with open(os.path.join(self.output_path, out_name), "w", encoding="utf-8") as f:
                json.dump(result, f, indent = 2)
        return result
    
    def main(self, scen_a, scen_b):
        dfs = self._process_data()
        result = self.explain_scenario_discrepancy(dfs, scen_a, scen_b, top_k_table=25, save_json=True, out_name=None)
        return result
    
# %% run module
if __name__ == '__main__':
    input_path = '../output_full/'
    feat_path = ''
    output_path = '../output_feature_importance/'
    flow_type_lst = ['corner', 'entrance', 'intersection']
    pred_obj = 'step_y'
    scen_a = 'intersection'
    scen_b = 'entrance'
    explainer = ScenDiffExplainer(input_path, feat_path, output_path, flow_type_lst, pred_obj)
    result = explainer.main(scen_a, scen_b)
    print(result)
# %%
