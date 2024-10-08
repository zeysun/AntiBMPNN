#!/bin/bash
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" == "AntiBMPNN" ]; then
    conda create -n mlfold python=3.11 numpy pandas torch peptides scikit-learn tqdm&& conda activate mlfold
    mkdir antibmpnn_model_weights
    cd antibmpnn_model_weights
    wget https://zenodo.org/records/13387792/files/model_weights.zip
    unzip model_weights.zip 
    rm model_weights.zip
    cd ..
    echo "Initialization done. You can try example_scripts.sh."
else
    echo "Current diractory is not AntiBMPNN, please download the git repo."
fi
