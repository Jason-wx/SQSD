import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def add_response_len(ds, tokenizer, response_col="response", num_proc=None):
    """Calculates the token length of the text."""
    def _batch_len_fn(batch):
        texts = [t if isinstance(t, str) else "" for t in batch[response_col]]
        enc = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        return {"response_len": [len(ids) for ids in enc["input_ids"]]}
    return ds.map(_batch_len_fn, batched=True, desc="Adding response_len", num_proc=num_proc)

def sample_fixed_window_and_plot(dataset, column, num_subsets=5, window_size=100, save_path=None, dpi=300):
    """Samples equidistant fixed windows and plots the distribution."""
    if column not in dataset.column_names:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {dataset.column_names}")

    print(f"\nSorting and Sampling by '{column}'...")
    # Sort in descending order
    sorted_dataset = dataset.sort(column, reverse=True)
    all_values = np.array(sorted_dataset[column])
    total_rows = len(dataset)

    # Calculate sampling start indices
    start_indices = np.linspace(0, total_rows - window_size, num_subsets, dtype=int)
    
    subsets = []
    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(total_rows), all_values, color='lightgray', label='Overall Distribution', alpha=0.8)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_subsets))

    for i, start in enumerate(start_indices):
        end = start + window_size
        subset = sorted_dataset.select(range(start, end))
        subsets.append(subset)
        
        subset_vals = all_values[start:end]
        plt.plot(np.arange(start, end), subset_vals, color=colors[i], linewidth=2.5)
        plt.axvspan(start, end, color=colors[i], alpha=0.15)
        
        # Automatically calculate annotation position
        label_text = f"S{i+1}: Idx {start}-{end}\nVal: {subset_vals[0]:.2f}→{subset_vals[-1]:.2f}"
        plt.annotate(label_text, xy=(start + window_size/2, subset_vals.mean()), 
                     xytext=(0, 40), textcoords='offset points', ha='center',
                     arrowprops=dict(arrowstyle='->', lw=0.5),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.9), fontsize=9)

    plt.title(f'Stratified Sampling Distribution (Column: {column})', fontsize=14)
    plt.xlabel('Rank (Descending)')
    plt.ylabel(f'Value of {column}')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Distribution plot saved to: {save_path}")
    
    return subsets

def main():
    parser = argparse.ArgumentParser()
    # Basic Parameters
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--danger_pt", type=str, required=True)
    parser.add_argument("--safe_pt", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    # Sampling Parameters
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--sample_column", type=str, default="Danger-Safe", help="Specifies the reference column for sampling")
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--num_subsets", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=1000)
    args = parser.parse_args()
    
    print(">>> Danger PT Path:", args.danger_pt)
    print(">>> Safe PT Path:", args.safe_pt)

    # 1. Loading and Cleaning
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = add_response_len(dataset, tokenizer)
    
    # 2. Injecting Scores
    danger_scores = torch.load(args.danger_pt, map_location='cpu')
    safe_scores = torch.load(args.safe_pt, map_location='cpu')
    dataset = dataset.add_column("Danger_Proj", danger_scores)
    dataset = dataset.add_column("Safe_Proj", safe_scores)
    
    # 3. Calculating Delta and Filtering
    dataset = dataset.map(lambda x: {"Danger-Safe": x["Danger_Proj"] - x["Safe_Proj"]})
    dataset = dataset.filter(lambda x: x["Danger_Proj"] != 0)
    
    # Saving Main File
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    dataset.to_json(args.output_file, force_ascii=False)
    print(f"\n[Step 1-5 Complete] Cleaned dataset saved. Rows: {len(dataset)}")

    # 4. Executing Sampling
    if args.do_sample:
        sample_dir = args.sample_dir or os.path.join(os.path.dirname(args.output_file), "samples")
        plot_path = os.path.join(sample_dir, f"sample_by_{args.sample_column}.png")
        
        subsets = sample_fixed_window_and_plot(
            dataset, 
            column=args.sample_column, 
            num_subsets=args.num_subsets, 
            window_size=args.window_size, 
            save_path=plot_path
        )
        
        for i, sub in enumerate(subsets):
            sub_file = os.path.join(sample_dir, f"sample_{i+1}.jsonl")
            sub.to_json(sub_file, force_ascii=False)
            print(f"Subset {i+1} (sorted by {args.sample_column}) saved to: {sub_file}")

if __name__ == "__main__":
    main()