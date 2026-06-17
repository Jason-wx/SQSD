#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

PYTHON_SCRIPT_PATH="./code/projection_grad_vector_scale_qwen3.py"
MODEL_PATH="./models/Qwen3-8B"
LORA_PATH="/Qwen3_8b_ckpt/PKU-SafeRlhf-10k-safer_5e-6/checkpoint-7000"
DATA_FILE="./data/Dolly/databricks-dolly-15k.jsonl"

SCALING_OVERRIDE=1 # alpha = scaling_overried/2
# outputdir
OUTPUT_DIR="./ProjScore/qwen3/SafeProj_7000_1"
mkdir -p $OUTPUT_DIR

# Data Field Name
FIELD_INSTRUCTION="instruction"
FIELD_RESPONSE="response"
FIELD_INPUT="context"

python "$PYTHON_SCRIPT_PATH" \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --scaling_override $SCALING_OVERRIDE \
    --datafile "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --metric "cosine"   \
    --instruction_field "$FIELD_INSTRUCTION" \
    --response_field "$FIELD_RESPONSE" \
    --input_field "$FIELD_INPUT"
