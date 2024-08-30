# AntiBMPNN
   To train/retrain AntiBMPNN clone this GitHub repo.
   Note: Before training, ensure your GPU driver has been installed properly.<br>
   The training data used to train AntiBMPNN can be downloaded here: `https://zenodo.org/records/13387792/files/training_set.zip`.<br>
   The dataset structure is the same as ProtenMPNN\`s and detailed info can be found here: `https://github.com/dauparas/ProteinMPNN/blob/main/training/README.md`.<br>
   Briefly, there are serval file types in the training set:<br>
```
PDBID_CHAINID.pt - contains CHAINID chain from PDBID
PDBID.pt         - metadata and information on biological assemblies
list.csv:
   CHAINID    - chain label, PDBID_CHAINID
   DEPOSITION - deposition date
   RESOLUTION - structure resolution
   HASH       - unique 6-digit hash for the sequence
   CLUSTER    - sequence cluster the chain belongs to (clusters were generated at seqID=30%)
   SEQUENCE   - reference amino acid sequence
valid_clusters.txt - clusters used for validation
test_clusters.txt - clusters used for testing
```

Code organization:<br>
* `Training.py` - the main script to train the model
* `model_utils_Antibody.py` - utility functions and classes for the model
* `utils_Antibody.py` - utility functions and classes for data loading
* `training_output/` - model training outputs

-----------------------------------------------------------------------------------------------------

The training parameters can be edited in `Training.py`, i.e. the lines with the comment "# Training parameters" at the end.<br>

-----------------------------------------------------------------------------------------------------
For example to make a python environment for training:<br>
` conda create -n mlfold python=3.11 numpy pandas torch peptides scikit-learn tqdm&& conda activate mlfold`

-----------------------------------------------------------------------------------------------------

Models provided for the AntiBMPNN were trained with flags:<br>
* `antibmpnn_000.pt` - `--num_neighbors 48 --backbone_noise 0.00 --num_epochs 200`
* `antibmpnn_010.pt` - `--num_neighbors 48 --backbone_noise 0.10 --num_epochs 200`
* `antibmpnn_020.pt` - `--num_neighbors 48 --backbone_noise 0.20 --num_epochs 200`
* `antibmpnn_030.pt` - `--num_neighbors 48 --backbone_noise 0.30 --num_epochs 200`
* `antibmpnn_040.pt` - `--num_neighbors 48 --backbone_noise 0.40 --num_epochs 200`
* `antibmpnn_050.pt` - `--num_neighbors 48 --backbone_noise 0.50 --num_epochs 200`
* `antibmpnn_060.pt` - `--num_neighbors 48 --backbone_noise 0.60 --num_epochs 200`
* `antibmpnn_070.pt` - `--num_neighbors 48 --backbone_noise 0.70 --num_epochs 200`
* `antibmpnn_080.pt` - `--num_neighbors 48 --backbone_noise 0.80 --num_epochs 200`
* `antibmpnn_090.pt` - `--num_neighbors 48 --backbone_noise 0.90 --num_epochs 200`
* `antibmpnn_100.pt` - `--num_neighbors 48 --backbone_noise 1.00 --num_epochs 200`
-----------------------------------------------------------------------------------------------------
