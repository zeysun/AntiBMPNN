#!/bin/bash

# Load environment variables
source activate mlfold

# Input job information
pdb_file_path="../inputs/HuJ3/"  
CHAINS_TO_DESIGN="B"
DESIGN_ONLY_POSITIONS="102 103 104 105 106"
THEME=$(date +"%m%d")_"example_design_fixed_positions"

# Define input and output directories
AB_NAME=$(basename "$pdb_file_path" .pdb)
INPUT_DIR="../inputs/${AB_NAME}/"
OUTPUT_DIR="./${THEME}_${AB_NAME}"

# Create an output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Define file paths
PATH_PARSED_CHAINS="$OUTPUT_DIR/parsed_pdbs.jsonl"
PATH_ASSIGNED_CHAINS="$OUTPUT_DIR/assigned_pdbs.jsonl"
PATH_FIXED_POSITIONS="$OUTPUT_DIR/fixed_pdbs.jsonl"

# Run preprocessing scripts
echo "Preprocessing..."
python ../helper_scripts/parse_multiple_chains.py --input_path=$INPUT_DIR --output_path=$PATH_PARSED_CHAINS
python ../helper_scripts/assign_fixed_chains.py --input_path=$PATH_PARSED_CHAINS --output_path=$PATH_ASSIGNED_CHAINS --chain_list "$CHAINS_TO_DESIGN"
python ../helper_scripts/make_fixed_positions_dict.py --input_path=$PATH_PARSED_CHAINS --output_path=$PATH_FIXED_POSITIONS --chain_list "$CHAINS_TO_DESIGN" --position_list "$DESIGN_ONLY_POSITIONS" --specify_non_fixed

echo "Running AntiBMPNN..."
python ../Running_AntiBMPNN_run.py \
    --jsonl_path $PATH_PARSED_CHAINS \
    --chain_id_jsonl $PATH_ASSIGNED_CHAINS \
    --fixed_positions_jsonl $PATH_FIXED_POSITIONS \
    --out_folder $OUTPUT_DIR \
    --model_name "antibmpnn_000" \
    --num_seq_per_target 1000 \
    --sampling_temp "0.1" \
    --batch_size 10 \
    --backbone_noise 0


read -r -a positions <<< "$DESIGN_ONLY_POSITIONS"
first_value=${positions[0]}
last_value=${positions[-1]}
fa_file_dir="${OUTPUT_DIR}/seqs/"
echo ${fa_file_dir}
echo ${first_value}
echo ${last_value}
python ../helper_scripts/parsing_design_result.py --dir ${fa_file_dir} --start ${first_value} --end ${last_value}

echo "Completed!"