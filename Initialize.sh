#!/bin/bash
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" == "AntiBMPNN" ]; then
    conda env create -f environment.yaml
    mkdir antibmpnn_model_weights
    cd antibmpnn_model_weights
    wget -O model_weights.zip https://zenodo.org/records/13381914/files/model_weights.zip
    unzip model_weights.zip && rm model_weights.zip
    cd ..
    echo "Initialization done. You can try example_scripts.sh."
else
    echo "Current diractory is not AntiBMPNN, please download the git repo."
fi
