#!/bin/bash

# --- Configuration ---
MODELS_DIR="../../ab/nn/nn"
MODEL_PATTERN="ast*.py"
CONFIG_PREFIX="img-classification_cifar-10_acc_"
EXTRA_FLAGS="-e 1 -t 10 --min_batch_binary_power 9 -f norm_256_flip"

# --- NEW: Centralized Logging Configuration ---
# All logs will be placed in this directory.
LOG_DIR="mutation_tracking_logs"
# This file tracks successfully completed models, allowing the script to be resumable.
PROCESSED_LOG_FILE="${LOG_DIR}/processed_models.log" 
# This file contains the detailed terminal output from the training sessions.
DETAILED_LOG_FILE="${LOG_DIR}/training_session_$(date +%Y-%m-%d_%H-%M-%S).log"

# --- Colors for better terminal output ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m' # For skipping
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Script Logic ---

# --- NEW: Ensure the log directory exists before we start ---
mkdir -p "$LOG_DIR"
echo -e "${BLUE}Log files will be stored in the '${LOG_DIR}' directory.${NC}"

log_and_print() {
    echo -e "$1" | tee -a "$DETAILED_LOG_FILE"
}

# Expand the tilde '~' to the full home directory path
EVAL_MODELS_DIR=$(eval echo "$MODELS_DIR")

# --- RESUME LOGIC: Load processed models into memory for fast checking ---
# Create the processed log file if it doesn't exist
touch "$PROCESSED_LOG_FILE"
# Use an associative array for efficient lookups.
declare -A processed_models
while IFS= read -r line; do
    processed_models["$line"]=1
done < "$PROCESSED_LOG_FILE"

# Get a list of all potential model files
shopt -s nullglob
model_files=("$EVAL_MODELS_DIR"/$MODEL_PATTERN)
total_models=${#model_files[@]}

if [ $total_models -eq 0 ]; then
    echo -e "${RED}Error: No model files found matching '$MODEL_PATTERN' in '$EVAL_MODELS_DIR'.${NC}"
    exit 1
fi

# Start logging session
echo "" > "$DETAILED_LOG_FILE" # Start a fresh detailed log for this session
log_and_print "${BLUE}====================================================${NC}"
log_and_print "${BLUE}Starting model training process at $(date)${NC}"
log_and_print "${BLUE}Found $total_models total models matching pattern: $MODEL_PATTERN${NC}"
log_and_print "${BLUE}${YELLOW}Found ${#processed_models[@]} previously processed models to skip.${NC}"
log_and_print "${BLUE}Detailed output will be saved to: $DETAILED_LOG_FILE${NC}"
log_and_print "${BLUE}====================================================${NC}"

current_model_num=0
for model_file_path in "${model_files[@]}"; do
    current_model_num=$((current_model_num + 1))
    
    filename=$(basename "$model_file_path")
    model_name="${filename%.py}"

    # --- THE CORE RESUME CHECK ---
    if [[ -v processed_models[$model_name] ]]; then
        # The model name exists in our 'processed' list, so we skip it.
        echo -e "${YELLOW}[$current_model_num/$total_models] SKIPPING model: ${model_name} (already processed)${NC}"
        continue # Move to the next model in the loop
    fi

    # If we reach here, the model has not been processed yet.
    log_and_print "\n${GREEN}[$current_model_num/$total_models] PROCESSING model: ${model_name}${NC}"
    
    full_command="python -m ab.nn.train -c ${CONFIG_PREFIX}${model_name} ${EXTRA_FLAGS}"
    log_and_print "--> Executing: ${full_command}"
    
    eval $full_command 2>&1 | tee -a "$DETAILED_LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        # SUCCESS! Log it to the processed file so we don't run it again.
        log_and_print "${GREEN}--> SUCCESS: Model ${model_name} finished.${NC}"
        echo "$model_name" >> "$PROCESSED_LOG_FILE"
    else
        # FAILURE! Do not log it, so it will be retried on the next run.
        log_and_print "${RED}ERROR: Model ${model_name} failed. It will be retried on the next run.${NC}"
    fi
done

log_and_print "\n${BLUE}====================================================${NC}"
log_and_print "${BLUE}Script run finished at $(date)${NC}"
log_and_print "${BLUE}====================================================${NC}"
