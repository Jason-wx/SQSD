#!/bin/bash
python ./safe_eval/safescore.py \
    --data_dir "./output/qwen3/Beaver-SQDS/generate_file" \
    --file_prefix "Beaver-Safe-1000-" \
    --num_files 5 \
    --batch_size 16 \
    --device 0

python ./safe_eval/guard3_1.py \
    --data_dir "./output/qwen3/Beaver-SQDS/generate_file" \
    --file_prefix "Beaver-Safe-1000-" \
    --num_files 5 \
    --batch_size 16 \
    --device 0
    