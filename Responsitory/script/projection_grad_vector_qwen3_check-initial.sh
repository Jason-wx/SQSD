#!/usr/bin/env bash

# Select GPU device
export CUDA_VISIBLE_DEVICES="0"

# Path to the base model and the Python execution script
PYTHON_SCRIPT_PATH="./code/projection_grad_vector_check-initial_qwen3.py"
MODEL_PATH="./models/Qwen3-8B"
DATA_FILE="./data/Dolly/databricks-dolly-15k.jsonl"

# Path to the target Task Vector LoRA
DIRECTION_PATH="./models/Direction_qwen3/danger/beavertails_unsafe_random3000_5e-6"
INITIAL_PATH="./models/Initial_qwen3/dolly_subset_5000/checkpoint-5850"

# Output directory for the results
OUTPUT_DIR="./ProjScore/qwen3/DangerProj_Beaver"
mkdir -p "$OUTPUT_DIR"

FIELD_INSTRUCTION="instruction"
FIELD_RESPONSE="response"
FIELD_INPUT="context"

# --- Execute Python Script ---
python "$PYTHON_SCRIPT_PATH" \
    --model_path "$MODEL_PATH" \
    --TaskVector_path "$DIRECTION_PATH" \
    --initial_path "$INITIAL_PATH" \
    --datafile "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --metric "cosine" \
    --instruction_field "$FIELD_INSTRUCTION" \
    --response_field "$FIELD_RESPONSE" \
    --input_field "$FIELD_INPUT
