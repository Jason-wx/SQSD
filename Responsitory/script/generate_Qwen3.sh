#!/bin/bash

# Configuration Paths
OUTPUT_BASE_DIR="./output/qwen3/Beaver-SQDS/generate_file"
EVAL_DATA_PATH="./safe_eval/bench/catqa_english.json"
GENERATION_SCRIPT="./inference/generation_qwen3.py"
Lora_dir="./output/qwen3/Beaver-SQDS/ckpt"

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Set GPU Device
export CUDA_VISIBLE_DEVICES=0

# Iterate through each LoRA checkpoint
for lora_path in "$Lora_dir"/*; do
    lora_name=$(basename "$lora_path")
    
    # Construct output file path
    output_file="${OUTPUT_BASE_DIR}/${lora_name}.jsonl"
    
    echo "=========================================="
    echo "Processing: $lora_name"
    echo "LoRA Path: $lora_path"
    echo "Output File: $output_file"
    echo "=========================================="
    
    # Check if the LoRA path exists
    if [ ! -d "$lora_path" ]; then
        echo "Warning: LoRA path does not exist: $lora_path"
        echo "Skipping this model..."
        continue
    fi
    
    # Execute generation command
    python "$GENERATION_SCRIPT" \
        --model_path "./model/Qwen3-8B" \
        --lora_path "$lora_path" \
        --output_path "$output_file" \
        --is_think no \
        --eval_data_path "$EVAL_DATA_PATH"
    
    # Check command execution result
    if [ $? -eq 0 ]; then
        echo "✓ $lora_name processing complete"
        echo "Output saved to: $output_file"
    else
        echo "✗ $lora_name processing failed"
    fi
    
    echo ""
done

echo "=========================================="
echo "All LoRA models processed!"
echo "Output Directory: $OUTPUT_BASE_DIR"
echo "=========================================="