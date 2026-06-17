#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import logging

# --- Now safe to import torch and other libraries ---
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import gc
from typing import Dict, Literal, Tuple, List, Optional, Union
import fire  # Import fire library
import json
from utils import get_lora_model
from utils_sft import preprocess, custom_data_collator, preprocess_qwen3

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

# cuda:0, bf16
def load_and_merge_lora(base_model_path: str, lora_path: str) -> Dict[str, torch.Tensor]:
    # (Logic same as previous version, depends on "cuda:0")
    logger.info("Calculating target task vector (B@A)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("TaskVector will load in CPU, its slow")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        lora_params = {
            k: v 
            for k, v in lora_model.state_dict().items() 
            if "lora" in k
        }
    finally:
        if 'base_model' in locals(): del base_model
        if 'lora_model' in locals(): del lora_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    submodule_tensors = defaultdict(dict)
    for name, tensor in lora_params.items():
        if ".lora_A." in name:
            submodule_name = name.split(".lora_A.")[0]
            submodule_tensors[submodule_name]['A'] = tensor
        elif ".lora_B." in name:
            submodule_name = name.split(".lora_B.")[0]
            submodule_tensors[submodule_name]['B'] = tensor
    merged_weights = {}
    for submodule_name, tensors in submodule_tensors.items():
        if 'A' in tensors and 'B' in tensors:
            delta_W = tensors['B'] @ tensors['A']
            delta_W = delta_W.to(torch.bfloat16)  # Convert all to bf16
            merged_weights[submodule_name] = delta_W
        else:
            logger.warning(f"Submodule '{submodule_name}' missing A or B matrix, skipped.")
    logger.info(f"Target task vector calculation complete. Found {len(merged_weights)} modules.")
    return merged_weights

def sum_dict_values(data_dict: Dict[str, Union[int, float]]) -> float:
    # (Logic same as previous version)
    return sum(data_dict.values())

# Perform this operation on the device of lora_delta_1
def calculate_lora_delta_metrics(
    lora_delta_1: Dict[str, torch.Tensor],
    lora_delta_2: Dict[str, torch.Tensor],
    metric: Literal["cosine", "l2", "l1", "dot", "projection"] = "cosine"
) -> Dict[str, float]:
    supported_metrics = ["cosine", "l2", "l1", "dot", "projection"]
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric: '{metric}'. Optional values: {supported_metrics}")
    results = {}
    for key, tensor1 in lora_delta_1.items():
        if key not in lora_delta_2:
            continue
        tensor2 = lora_delta_2[key]
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)
        v1 = tensor1.flatten().float()
        v2 = tensor2.flatten().float()
        score = 0.0
        if metric == "cosine":
            score = F.cosine_similarity(v1, v2, dim=0).item()
        elif metric == "l2":
            score = torch.linalg.norm(v1 - v2).item()
        elif metric == "l1":
            score = torch.linalg.norm(v1 - v2, ord=1).item()
        elif metric == "dot":
            score = torch.dot(v1, v2).item()
        elif metric == "projection":
            dot_product = torch.dot(v1, v2)
            norm_v2 = torch.linalg.norm(v2)
            score = (dot_product / norm_v2).item() if norm_v2 > 1e-9 else 0.0
        results[key] = score
    return results

# Execute on CUDA
def compute_task_vector_delta_step(
    lora_state: Dict[str, torch.Tensor],
    grad_state: Dict[str, torch.Tensor],
    lr: float,
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    # (Logic same as previous version)
    def _split_lora_key(name: str):
        k = name.replace(".default", "")
        if k.endswith(".lora_A.weight"): return k[: -len(".lora_A.weight")], "A"
        if k.endswith(".lora_B.weight"): return k[: -len(".lora_B.weight")], "B"
        if k.endswith(".lora_down.weight"): return k[: -len(".lora_down.weight")], "A"
        if k.endswith(".lora_up.weight"): return k[: -len(".lora_up.weight")], "B"
        return None, None
    lora_by_sub = defaultdict(dict)
    grad_by_sub = defaultdict(dict)
    for k, v in lora_state.items():
        sub, tag = _split_lora_key(k)
        if sub is not None: lora_by_sub[sub][tag] = v
    for k, g in grad_state.items():
        sub, tag = _split_lora_key(k)
        if sub is not None: grad_by_sub[sub][tag] = g
    merged_delta: Dict[str, torch.Tensor] = {}
    for sub in lora_by_sub.keys():
        A = lora_by_sub[sub].get("A", None)
        B = lora_by_sub[sub].get("B", None)
        gA = grad_by_sub.get(sub, {}).get("A", None)
        gB = grad_by_sub.get(sub, {}).get("B", None)
        if A is None or B is None or gA is None or gB is None: continue
        if not (B.dim() == 2 and A.dim() == 2 and gB.shape == B.shape and gA.shape == A.shape):
            raise ValueError(f"[{sub}] unexpected shapes: B{tuple(B.shape)}, A{tuple(A.shape)}, gB{tuple(gB.shape)}, gA{tuple(gA.shape)}")
        delta_v = -(lr * ((gB @ A) + (B @ gA)))    # Core code
        if scale != 1.0:
            delta_v = delta_v * scale
        merged_delta[sub] = delta_v
    return merged_delta



def main(
    model_path: str,
    lora_path: str,
    scaling_override: int,
    datafile: str,
    output_dir: str,
    metric: Literal["cosine", "l2", "l1", "dot", "projection"] = "cosine",
    instruction_field: str = "instruction",
    response_field: str = "response",
    input_field: str = "context",
):
    """
    Main execution function.
    (Parameter documentation same as previous version)
    """
    
    logger.info(f"Script started. torch will use os.environ['CUDA_VISIBLE_DEVICES']='{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    logger.info("Run parameters:")
    for k, v in locals().items():
        logger.info(f"  --{k}: {v}")

    # Python script internally hardcoded to use "cuda:0".
    # Which physical GPU "cuda:0" maps to is determined by the CUDA_VISIBLE_DEVICES environment variable.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        logger.error("CUDA not detected! Please check CUDA_VISIBLE_DEVICES settings or PyTorch installation.")
        return
        
    logger.info(f"Using logical device: {device} (mapped to physical GPU: {torch.cuda.get_device_name(0)})")

    # --- Step 1: Calculate target task vector (B@A). This is also the anchor direction ---
    target_task_vector = load_and_merge_lora(model_path, lora_path)

    # --- Step 2: Load model for gradient calculation ---
    model, tokenizer = get_lora_model(
        model_path=model_path,
        lora_path=lora_path
    )
    for name, param in model.named_parameters():
            param.to(torch.bfloat16)
            if "lora" in name:
                param.requires_grad = True
    
    # --- Step 3: Modify lora_alpha ---
    try:
        from peft.tuners.lora import LoraLayer

        SCALING = scaling_override   # Desired injection strength; 0.0 = disable LoRA injection
        SYNC_ALPHA = False  # If True, sync lora_alpha to scaling * r (usually not needed)

        def _to_float(x):
            try:
                if hasattr(x, "item"):
                    return float(x.item())
                return float(x)
            except Exception:
                return None

        def _pick_single_adapter(m):
            """
            Used only in "single adapter" scenarios:
            - If lora_A/lora_B is a dict and has only one key, take that key;
            - Otherwise use active_adapter; finally fallback to 'default'
            """
            for attr in ("lora_A", "lora_B"):
                if hasattr(m, attr) and isinstance(getattr(m, attr), dict) and len(getattr(m, attr)) == 1:
                    return next(iter(getattr(m, attr).keys()))
            
            # --- Modification start ---
            # Get active_adapter, which could be a string or a list
            adapter_name_or_list = getattr(m, "active_adapter", "default")
            
            # Ensure a string is returned
            if isinstance(adapter_name_or_list, (list, tuple)):
                if len(adapter_name_or_list) > 0:
                    # If list, take the first element
                    return adapter_name_or_list[0]
                else:
                    # If empty list, return 'default'
                    return "default"
            
            # If it's already a string (or 'default'), return directly
            return str(adapter_name_or_list)
            # --- Modification end ---

        num_layers = 0
        # Modify model scaling
        for m in model.modules():
            if not isinstance(m, LoraLayer):
                continue

            adp = _pick_single_adapter(m)

            # ---- Set injection strength directly (Critical) ----
            if hasattr(m, "scaling"):
                if isinstance(m.scaling, dict):
                    m.scaling[adp] = float(SCALING)
                else:
                    m.scaling = float(SCALING)
            if hasattr(m, "scale"):
                if isinstance(m.scale, dict):
                    m.scale[adp] = float(SCALING)
                else:
                    m.scale = float(SCALING)

            # ---- (Optional) Synchronize lora_alpha = scaling * r ----
            if SYNC_ALPHA and hasattr(m, "r") and hasattr(m, "lora_alpha"):
                r_val = m.r.get(adp, None) if isinstance(m.r, dict) else m.r
                r = _to_float(r_val)
                if r is not None and r > 0:
                    new_alpha = float(SCALING) * r
                    if isinstance(m.lora_alpha, dict):
                        m.lora_alpha[adp] = new_alpha
                    else:
                        m.lora_alpha = new_alpha

            num_layers += 1

        logging.info(f"[LoRA] Single adapter mode: Set injection strength for {num_layers} LoRA layers to {SCALING}")
    except Exception as e:
        logging.exception(f"Failed to set LoRA scaling: {e}")


    # This holds the LoRA parameters of the current model
    lora_params_cache = {
        name: p.detach().clone().to(torch.bfloat16)
        for name, p in model.named_parameters() if "lora" in name
    }
    logger.info("Cached current LoRA parameters for gradient calculation.")

    # --- Step 4: Load and process dataset ---
    logger.info(f"Loading dataset from {datafile}...")
    dataset = load_dataset("json", data_files=datafile, split="train")
    
    logger.info("Preprocessing dataset...")
    logger.info("Using preprocess to handle data")
    dataset_processed = dataset.map(
        lambda example: preprocess(
            example,
            tokenizer, 
            instruction_id=instruction_field, 
            response_id=response_field,
            input_id=input_field
        ),
        remove_columns=dataset.column_names,
        desc="Processing dataset"
    )

    dataloader = DataLoader(
        dataset_processed,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda features: custom_data_collator(features, tokenizer),
    )

    # --- Step 5: Loop to calculate projections ---
    projection_scores = []
    logger.info("Starting projection calculation loop...")
    logger.info(f"the metric is {metric}")

    
    for step, batch in enumerate(tqdm(dataloader, desc="Calculating Projections")):
        
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        loss = output.loss
        del output

        loss.backward()
        del loss

        with torch.no_grad():
            grad_dict = {
                k: p.grad.detach().to("cuda:0", non_blocking=True).to(torch.bfloat16)
                for k, p in model.named_parameters() if p.grad is not None
            }
            model.zero_grad(set_to_none=True)

            delta_step = compute_task_vector_delta_step(
                lora_params_cache, 
                grad_dict, 
                lr=1.0
            )
            
            # Computation occurs on CPU
            projection_metrics = calculate_lora_delta_metrics(
                lora_delta_1=delta_step,
                lora_delta_2=target_task_vector,
                metric=metric
            )

            total_score = sum_dict_values(projection_metrics)
            if hasattr(total_score, "detach"):
                total_score = total_score.detach().cpu().item()
            projection_scores.append(total_score)

        del delta_step, grad_dict
        # Save results every 1000 steps
        if (step+1)%1000 == 0:
            score_filename = os.path.join(output_dir, f"scores_{step+1}.pt")
            torch.save(projection_scores, score_filename)

    # --- Step 6: Save final results ---
    logger.info("Projection calculation finished.")
    output_file = os.path.join(output_dir, f"scores_{step+1}.pt")
    if len(projection_scores) != len(dataset):
        logger.error(f"Score count ({len(projection_scores)}) does not match dataset size ({len(dataset)})!")
    else:
        logger.info(f"Saving dataset to {output_file}...")

        try:
            torch.save(projection_scores, output_file)
            logger.info(f"✅ Successfully saved to {output_file}")

        except Exception as e:
            logger.error(f"❌ Error occurred while saving to .pt file: {e}")


if __name__ == "__main__":
    fire.Fire(main)