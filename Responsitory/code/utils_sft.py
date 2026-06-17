from typing import Optional
import torch

def preprocess(example, tokenizer, instruction_id: str, response_id: str, input_id: Optional[str] = None):
    """
    Preprocess a single sample.
    
    Args:
        example: Input sample containing instruction and response
        tokenizer: Tokenizer to use for encoding
        instruction_id: Key for instruction field in the example
        response_id: Key for response field in the example
        input_id: Optional key for input/context field in the example
    
    Returns:
        dict: Processed encoding results
    """
    # Extract instruction, context and response
    instruction = example[instruction_id]
    output_text = example[response_id] + "<|im_end|>"

    # Build conversation messages
    if input_id and input_id in example and example[input_id].strip():
        input_text = example[input_id]
        messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    else:
        messages = [{"role": "user", "content": instruction}]

    # Use tokenizer.apply_chat_template to generate conversation format
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,  # Return string instead of token ids
            add_generation_prompt=True  # Add generation prompt
        )
    except Exception as e:
        # Fallback if tokenizer doesn't support chat_template
        print(f"Warning: {e}")
        if input_id and input_id in example and example[input_id].strip():
            input_text = example[input_id]
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Combine input and output text
    full_text = prompt + output_text
    
    # Encode full text
    encoded = tokenizer(
        full_text,
        truncation=True, 
        padding=False,  # Padding will be done in DataLoader
        max_length=1024,  
        return_tensors=None  # Return list, not tensor
    )
    
    # Encode prompt part for calculating labels
    prompt_encoded = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        max_length=2048,
        return_tensors=None
    )
    
    # Create labels, only calculate loss for response part
    labels = encoded["input_ids"].copy()
    prompt_length = len(prompt_encoded["input_ids"])
    
    # Set prompt part labels to -100 (ignore in loss calculation)
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }

def preprocess_qwen3(example, tokenizer, instruction_id: str, response_id: str, input_id: Optional[str] = None):
    """
    Preprocess a single sample.
    
    Args:
        example: Input sample containing instruction and response
        tokenizer: Tokenizer to use for encoding
        instruction_id: Key for instruction field in the example
        response_id: Key for response field in the example
        input_id: Optional key for input/context field in the example
    
    Returns:
        dict: Processed encoding results
    """
    # Extract instruction, context and response
    instruction = example[instruction_id]
    output_text = example[response_id] + "<|im_end|>"

    # Build conversation messages
    if input_id and input_id in example and example[input_id].strip():
        input_text = example[input_id]
        messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    else:
        messages = [{"role": "user", "content": instruction}]

    # Use tokenizer.apply_chat_template to generate conversation format
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,  # Return string instead of token ids
            add_generation_prompt=True,  # Add generation prompt
            enable_thinking=False
        )
    except Exception as e:
        # Fallback if tokenizer doesn't support chat_template
        print(f"Warning: {e}")
        if input_id and input_id in example and example[input_id].strip():
            input_text = example[input_id]
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Combine input and output text
    full_text = prompt + output_text
    
    # Encode full text
    encoded = tokenizer(
        full_text,
        truncation=True, 
        padding=False,  # Padding will be done in DataLoader
        max_length=1024,  
        return_tensors=None  # Return list, not tensor
    )
    
    # Encode prompt part for calculating labels
    prompt_encoded = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        max_length=2048,
        return_tensors=None
    )
    
    # Create labels, only calculate loss for response part
    labels = encoded["input_ids"].copy()
    prompt_length = len(prompt_encoded["input_ids"])
    
    # Set prompt part labels to -100 (ignore in loss calculation)
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }


def custom_data_collator(features, tokenizer):
    """
    Custom data collator function to ensure proper padding and tensor format.
    
    Args:
        features: List of processed features
        tokenizer: Tokenizer for padding configuration
        
    Returns:
        dict: Batched and padded data
    """
    # Get maximum length
    max_length = max(len(f["input_ids"]) for f in features)
    
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        batch[key] = []
        
    for feature in features:
        # Padding to same length
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]
        
        # Calculate padding length needed
        pad_length = max_length - len(input_ids)
        
        if pad_length > 0:
            # Padding
            if tokenizer.padding_side == "right":
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [-100] * pad_length
            else:  # left padding
                input_ids = [tokenizer.pad_token_id] * pad_length + input_ids
                attention_mask = [0] * pad_length + attention_mask
                labels = [-100] * pad_length + labels
        
        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)
    
    # Convert to tensor
    for key in batch:
        batch[key] = torch.tensor(batch[key], dtype=torch.long)
    
    return batch