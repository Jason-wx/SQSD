#!/bin/bash

MODEL_PATH="./models/Qwen3-8B"
RAW_DATA="./data/Dolly/databricks-dolly-15k.jsonl"
DANGER_PT="./ProjScore/qwen3/DangerProj_Beaver/scores_15011.pt"
SAFE_PT="./ProjScore/qwen3/SafeProj_7000_1/scores_15011.pt"


BASE_OUT_DIR="./output/qwen3/Beaver-SQDS"
CLEANED_FILE="${BASE_OUT_DIR}/cleaned_dolly-15k.jsonl"
SAMPLE_DIR="${BASE_OUT_DIR}/data/5_1000"


python ./code/process_and_sample.py \
    --model_path "$MODEL_PATH" \
    --data_path "$RAW_DATA" \
    --danger_pt "$DANGER_PT" \
    --safe_pt "$SAFE_PT" \
    --output_file "$CLEANED_FILE" \
    --do_sample \
    --sample_column "Danger-Safe" \
    --sample_dir "$SAMPLE_DIR" \
    --num_subsets 5 \
    --window_size 1000

echo "Pipeline Finished!"