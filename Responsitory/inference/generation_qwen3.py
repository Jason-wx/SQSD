from datasets import load_dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from peft import PeftModel
import argparse
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Literal

# Qwen3 </think> token id (Official example is usually 151668)
END_THINK_TOKEN_ID = 151668


def get_model(model_path: str, lora_path: str = None):
    """
    Load Qwen3-8B (or other compatible models) and optional LoRA weights.
    """
    # Load tokenizer and model directly from model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    # If Qwen3 lacks a pad_token, use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, model_id=lora_path)

    return tokenizer, model


def generate(
    data_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    output_file: str,
    is_think: Literal["yes", "no"]
):
    """
    Use the model to generate responses in batches and write to a JSONL file.

    :param data_loader: PyTorch DataLoader providing batch data (e.g., {"Question": [...]})
    :param tokenizer: Transformers tokenizer
    :param model: Transformers model
    :param output_file: Output path for the JSONL file
    :param is_think: Mode flag ('yes' or 'no')
    """

    # 1. System Prompt (Optional)
    if is_think == "no":
        print("----------- Standard Generation Mode (enable_thinking=False) ---------")
        system_prompt = "You are a helpful assistant."
    elif is_think == "yes":
        print("----------- Thinking Generation Mode (enable_thinking=True) ---------")
        # For Qwen3, logic relies on the enable_thinking flag; the prompt can be simple
        system_prompt = "You are a helpful assistant"
    else:
        print("is_think must be 'yes' or 'no'")
        return None

    enable_thinking_flag = (is_think == "yes")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Processing data", leave=False):
                    # Assemble messages
                    messages_batch = [
                        [
                            # {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        for prompt in batch["Question"]
                    ]

                    # Use Qwen3 chat template and toggle thinking mode based on is_think
                    texts = [
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking_flag,  # Key toggle
                        )
                        for messages in messages_batch
                    ]

                    # Tokenize
                    inputs = tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(model.device)

                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False  # Greedy decoding
                    )

                    # Slice only the newly generated part
                    input_length = inputs["input_ids"].shape[1]
                    generated_ids = outputs[:, input_length:]  # [batch, new_len]
                    generated_ids_list = generated_ids.tolist()

                    # Write to JSONL
                    for prompt, out_ids in zip(batch["Question"], generated_ids_list):
                        if enable_thinking_flag:
                            # Split thinking / answer according to official examples
                            try:
                                # Find index of END_THINK_TOKEN_ID from the right
                                index = len(out_ids) - out_ids[::-1].index(END_THINK_TOKEN_ID)
                            except ValueError:
                                # No </think> token found; assume no explicit reasoning
                                index = 0

                            thinking_ids = out_ids[:index]
                            answer_ids = out_ids[index:]

                            thinking_text = tokenizer.decode(
                                thinking_ids,
                                skip_special_tokens=True
                            ).strip("\n")
                            answer_text = tokenizer.decode(
                                answer_ids,
                                skip_special_tokens=True
                            ).strip("\n")

                            json_record = {
                                "query": prompt,
                                "thinking": thinking_text,
                                "response": answer_text,
                            }
                        else:
                            # Standard mode: only take final answer
                            response_text = tokenizer.decode(
                                out_ids,
                                skip_special_tokens=True
                            ).strip("\n")

                            json_record = {
                                "query": prompt,
                                "response": response_text,
                            }

                        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

                    f.flush()

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="e.g., Qwen/Qwen3-8B or a local path")
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--eval_data_path",
        default="./safe_eval/bench/catqa_english.json",
        type=str
    )
    parser.add_argument(
        "--is_think",
        default="no",
        type=str,
        choices=["yes", "no"],
        help="Whether to enable thinking mode (maps to enable_thinking)"
    )
    args = parser.parse_args()

    print("Loading model..............")
    tokenizer, model = get_model(model_path=args.model_path, lora_path=args.lora_path)
    print("------------ Model is loaded ---------")

    eval_dataset = load_dataset("json", data_files=args.eval_data_path)
    dataloader = DataLoader(
        eval_dataset["train"],
        batch_size=32,
        shuffle=False
    )
    print("------------ Evaluation data is loaded ---------")

    generate(
        dataloader,
        tokenizer,
        model,
        output_file=args.output_path,
        is_think=args.is_think
    )

    print("Generation complete.")