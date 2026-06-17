from peft import (LoraConfig, TaskType, get_peft_model, PeftModel)
from transformers import AutoTokenizer, AutoModelForCausalLM
# from safe_rlhf.models import AutoModelForScore
import torch
import torch.nn.functional as F
import os
from typing import Dict, Literal, Tuple, List, Optional, Union
from collections import defaultdict

def load_and_merge_lora(base_model_path: str, lora_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads the base model and LoRA adapter, extracts LoRA parameters, and merges A and B matrices.

    This function encapsulates the following process:
    1. Loads the model and LoRA adapter to GPU from paths.
    2. Extracts all LoRA-related parameters (A and B matrices).
    3. Calculates the matrix product B @ A for each submodule.
    4. Returns a dictionary mapping submodule names to the merged weight delta matrices.

    Args:
        base_model_path (str): Path to the base model.
        lora_path (str): Path to the LoRA adapter weights.

    Returns:
        Dict[str, torch.Tensor]: A dictionary where keys are submodule names (without LoRA suffixes)
                                 and values are the results of $B \times A$ ($\Delta W$).
    """
    # --- Steps 1 & 2: Load model and extract LoRA parameters ---
    
    # Define device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not detected, running on CPU. This may be slow.")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16, 
        device_map=device
    )

    # Load LoRA adapter
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    # Extract LoRA parameters
    lora_params = {
        k: v 
        for k, v in lora_model.state_dict().items() 
        if "lora" in k
    }
    
    # Release model memory as we only need lora_params
    del base_model
    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 3: Merge A and B matrices ---

    # Group tensors by submodule name
    submodule_tensors = defaultdict(dict)
    for name, tensor in lora_params.items():
        # Handle common peft LoRA key formats (e.g., '...q_proj.lora_A.weight')
        if ".lora_A." in name:
            submodule_name = name.split(".lora_A.")[0]
            submodule_tensors[submodule_name]['A'] = tensor
        elif ".lora_B." in name:
            submodule_name = name.split(".lora_B.")[0]
            submodule_tensors[submodule_name]['B'] = tensor

    merged_weights = {}
    
    # Iterate through grouped tensors and compute B @ A
    for submodule_name, tensors in submodule_tensors.items():
        if 'A' in tensors and 'B' in tensors:
            lora_A = tensors['A']
            lora_B = tensors['B']
            
            # Matrix multiplication: $\Delta W = B \times A$
            delta_W = lora_B @ lora_A
            
            merged_weights[submodule_name] = delta_W
        else:
            print(f"Warning: Submodule '{submodule_name}' is missing matrix A or B, skipping.")
            
    return merged_weights

def merge_lora_dict(lora_dict) -> Dict[str, torch.Tensor]:
    """
    Merges LoRA weights from a given state dictionary into $B \times A$ deltas.
    """
    # Define device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not detected, running on CPU. This may be slow.")

    # Group tensors by submodule name
    submodule_tensors = defaultdict(dict)
    for name, tensor in lora_dict.items():
        if ".lora_A." in name:
            submodule_name = name.split(".lora_A.")[0]
            submodule_tensors[submodule_name]['A'] = tensor
        elif ".lora_B." in name:
            submodule_name = name.split(".lora_B.")[0]
            submodule_tensors[submodule_name]['B'] = tensor

    merged_weights = {}
    
    # Iterate through grouped tensors and compute B @ A
    for submodule_name, tensors in submodule_tensors.items():
        if 'A' in tensors and 'B' in tensors:
            lora_A = tensors['A']
            lora_B = tensors['B']
            
            # Matrix multiplication B @ A
            delta_W = lora_B @ lora_A
            merged_weights[submodule_name] = delta_W
        else:
            print(f"Warning: Submodule '{submodule_name}' is missing matrix A or B, skipping.")
            
    return merged_weights

def get_inital_model(model_path: str, is_lora: bool):
    """
    Initializes the model and tokenizer. Adds LoRA config if is_lora is True.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if is_lora:
        lora_r = 8
        lora_dropout = 0.05
        lora_alpha = 16
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=lora_dropout,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer
    else:
        return model, tokenizer
    
def get_lora_model(model_path, lora_path):
    """
    Loads a base model and wraps it with a pre-trained LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", truncation_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    lora_model = PeftModel.from_pretrained(model, model_id=lora_path)
    return lora_model, tokenizer

def get_reward_model(model_id):
    """
    Loads a reward model for scoring.
    """
    model = AutoModelForScore.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def sample_next_token_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    Samples the next token ID and log probability from logits given sampling parameters.
    """
    if logits.dim() == 3:
        next_token_logits = logits[:, -1, :]
    else:
        next_token_logits = logits[-1, :].unsqueeze(0)  # shape: (1, vocab_size)
    
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature
    
    if top_k > 0:
        values, _ = torch.topk(next_token_logits, top_k)
        min_value = torch.finfo(next_token_logits.dtype).min
        next_token_logits = torch.where(
            next_token_logits < values[..., -1, None],
            torch.tensor(min_value, device=next_token_logits.device),
            next_token_logits
        )
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        min_value = torch.finfo(next_token_logits.dtype).min
        next_token_logits = torch.where(
            indices_to_remove,
            torch.tensor(min_value, device=next_token_logits.device),
            next_token_logits
        )
    
    probs = F.softmax(next_token_logits, dim=-1)
    sampled_token_id = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)
    log_prob = torch.log(probs.gather(-1, sampled_token_id))  # shape: (1, 1)
    return sampled_token_id.squeeze(0), log_prob.squeeze(0)  # shape: (1,), (1,)

def generate_sequence_from_model(
    model,
    attention_mask,
    tokenizer,
    input_ids,
    max_new_tokens=20,
    temperature=1,
    top_k=0,
    top_p=0,
    device=None
):
    """
    Generates a sequence and calculates differentiable total log probabilities.
    """
    if device is None:
        device = input_ids.device
    
    with torch.no_grad():
        current_input_ids = input_ids.to(model.device)
        current_attention_mask = attention_mask.to(model.device)
        generated_token_ids = []  # Store token IDs only (ints), not tensors
        token_log_probs_float = []  # For logging purposes, no gradients
        
        for _ in range(max_new_tokens):
            model_output = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
            logits = model_output.logits
            
            next_token_id, log_prob = sample_next_token_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Save token ID as int to break the computation graph
            generated_token_ids.append(next_token_id.item())
            
            # For logging (fully detached)
            token_log_probs_float.append(log_prob.detach().item())
            
            # NOTE: We do not save log_prob tensors here to allow graph recycling.
            
            # Update sequence
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=-1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1
            )
    
    # Re-run forward pass to calculate total log probability with a clean computation graph
    if generated_token_ids:
        # Construct full sequence
        full_sequence = torch.cat([
            input_ids.squeeze(0), 
            torch.tensor(generated_token_ids, device=device)
        ], dim=0).unsqueeze(0)
        
        full_attention_mask = torch.ones_like(full_sequence)
        
        # Re-forward with gradients enabled
        with torch.set_grad_enabled(True):
            model_output = model(input_ids=full_sequence, attention_mask=full_attention_mask)
            logits = model_output.logits
            
            input_length = input_ids.size(1)
            token_log_probs = []
            
            for i, token_id in enumerate(generated_token_ids):
                # Get logits for the specific position
                position_logits = logits[0, input_length + i - 1, :]
                
                # Apply same sampling parameters
                if temperature != 1.0:
                    position_logits = position_logits / temperature
                
                if top_k > 0:
                    values, _ = torch.topk(position_logits, top_k)
                    min_value = torch.finfo(position_logits.dtype).min
                    position_logits = torch.where(
                        position_logits < values[-1],
                        torch.tensor(min_value, device=position_logits.device),
                        position_logits
                    )
                
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(position_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    min_value = torch.finfo(position_logits.dtype).min
                    position_logits = torch.where(
                        indices_to_remove,
                        torch.tensor(min_value, device=position_logits.device),
                        position_logits
                    )
                
                # Calculate log probability for this token (this has gradients)
                probs = F.softmax(position_logits, dim=-1)
                token_log_prob = torch.log(probs[token_id])
                token_log_probs.append(token_log_prob)
            total_log_prob = torch.stack(token_log_probs).sum()
    else:
        total_log_prob = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Build final sequence tensor
    if generated_token_ids:
        final_token_tensor = torch.cat([
            input_ids,
            torch.tensor(generated_token_ids, device=device).unsqueeze(0)
        ], dim=-1)
    else:
        final_token_tensor = input_ids
    
    return final_token_tensor, token_log_probs_float, total_log_prob