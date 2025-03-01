#!/bin/bash

# Set your API key and other parameters
API_BASE="https://openrouter.ai/api/v1"
API_KEY="YOUR_API_KEY"
MODEL_NAME="deepseek/deepseek-chat"
DATASET_NAME="question"
SPLIT="train"

DATA_PATH="datasets"
DATA_SAMPLE_CODE_PATH="datasets/result_code"

#Step 1: Generate tool samples using deep_tool_creation.py
echo "Generating tool samples for ${DATASET_NAME}..."
python "deep_tool_factory.py" \
    --api_key "${API_KEY}" \
    --api_base "${API_BASE}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --data_path "${DATA_PATH}" \
    --save_path "./results"
echo "Generating prompts and inference completed!"

# Step 2: Parse and execute the generated samples using code_execution.py
# echo "Parsing and executing generated samples for Dataset ${DATASET_NAME}..."
python "code_execution.py" \
    --json_file_path "./results/answer.json" \
    --dataset_name "${DATASET_NAME}" \
    --python_files_folder "${DATA_SAMPLE_CODE_PATH}" \
    --output_folder "${DATASET_NAME}"
echo "Code execution and result generation completed!"