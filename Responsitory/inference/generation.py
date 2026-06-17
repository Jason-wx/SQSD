from datasets import load_dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 
from torch.utils.data import DataLoader, Dataset, DistributedSampler 
import torch
from peft import PeftModel
import os
import argparse
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Literal

def get_model(model_path, lora_path):
    """
    Loads the model and tokenizer, applying LoRA if a path is provided.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"  # Set padding side to left for batch generation
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if lora_path is None:
        return tokenizer, model
    else:
        lora_model = PeftModel.from_pretrained(model, model_id=lora_path)
        return tokenizer, lora_model

def generate(
    data_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    output_file: str,
    is_think: Literal["yes", "no"]
):
    """
    Use the model to generate responses in batches and write them to a JSONL file.

    :param data_loader: PyTorch DataLoader providing batches (e.g., {"Question": [...]})
    :param tokenizer: Transformers tokenizer
    :param model: Transformers model
    :param output_file: Path to the output JSONL file
    :param is_think: Mode flag ('yes' for chain-of-thought or 'no' for direct answer)
    """
    
    # 1. Set System Prompt
    if is_think == "no":
        print("----------- Generation Mode ---------")
        system_prompt = "You are a helpful assistant."
    elif is_think == "yes":
        print("----------- Reasoning Generation Mode ---------")
        system_prompt = """
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
        The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.  \
        The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
        """
    else:
        print("is_think must be 'yes' or 'no'")
        return None

    # 2. Open file and start processing
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # Critical optimization: use torch.no_grad() for inference
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Processing data", leave=False):
                    
                    # Prepare chat template
                    messages_batch = [
                        [
                            # {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        for prompt in batch["Question"]
                    ]

                    # Convert to text format
                    texts = [
                        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        for messages in messages_batch
                    ]
                    
                    # Tokenize inputs
                    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

                    # Generate responses
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False  # Use greedy decoding
                    )

                    # --- Optimization: Extract response based on input length ---
                    # 1. Get input token length
                    #    inputs['input_ids'].shape[1] is the input length *after padding*
                    input_length = inputs['input_ids'].shape[1]

                    # 2. Decode only the newly generated part
                    #    outputs contains [input_tokens] + [generated_tokens]
                    #    We slice from input_length to get only [generated_tokens]
                    generated_texts = [
                        tokenizer.decode(seq[input_length:], skip_special_tokens=True)
                        for seq in outputs
                    ]
                    # --- End of optimization ---

                    # Write to JSONL
                    for prompt, generated_text in zip(batch["Question"], generated_texts):
                        json_record = {"query": prompt, "response": generated_text}
                        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                    
                    f.flush()  # Ensure immediate disk write

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str, required=True)
    parser.add_argument('--lora_path', default=None, type=str)
    parser.add_argument('--output_path', default=None, type=str, required=True)
    parser.add_argument('--eval_data_path', default="./safe_eval/bench/catqa_english.json", type=str)
    parser.add_argument('--is_think', default="no", type=str)
    args = parser.parse_args()

    print("Loading model..............")
    tokenizer, model = get_model(model_path=args.model_path, lora_path=args.lora_path)
    model = model.eval()
    print("------------ Model loaded ---------")

    eval_dataset = load_dataset("json", data_files=args.eval_data_path)
    dataloader = DataLoader(
        eval_dataset["train"],
        batch_size=32,
        shuffle=False
    )
    print("------------ Evaluation data loaded ---------")

    generate(dataloader, tokenizer, model, output_file=args.output_path, is_think=args.is_think)
    print("Generation complete.")