<p align="left">
<img width="800" src="https://github.com/zeysun/AntiBMPNN/blob/main/figures/banner.jpg">  
</p>  

###  This is the repo of AntiBMPNN project, for antibody sequence design.  

***  

## Protocol
<p align="center">
<img width="550" src="https://github.com/zeysun/AntiBMPNN/blob/main/figures/AntiBMPNN_1.jpg">
</p>



## Installation
### Requirements: 
#### Hardware Requirement:
AntiBMPNN can run on both **CPU** and **GPU**. For optimal performance, using an NVIDIA graphics card is recommended.  
#### System Requirement: Linux & Conda
A **Linux** based operating system is required.   
We have tested our code on both **Ubuntu** system and **[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)** (Click the link for installation instructions for Windows Subsystem for Linux (WSL).).   
  
**An [Anaconda python environment](https://www.anaconda.com/download) or [Miniconda python environment](https://docs.anaconda.com/miniconda/install/) is **required**.(Click the link for installation instructions for Anaconda/Miniconda.)**
- Python >= 3.11
- numpy
- pandas
- torch
- peptides
- scikit-learn
- tqdm
  
### 1. Obtaining AntiBMPNN Source Code
Download the AntiBMPNN repository:  
```sh
git clone https://github.com/zeysun/AntiBMPNN
```

### 2. One-step Installation (recommended way)
We provided an auto-install bash script. This script will automatically build a python environment via conda and download packages and model weights.
```sh
cd AntiBMPNN  
bash Initialize.sh
```
#### Manual Installation
If you have completed the one-step installation, you can skip the manual installation step. This section is provided for reference, just in case.
```sh
conda create -n mlfold python=3.11
conda activate mlfold
pip install numpy pandas torch peptides scikit-learn tqdm
```
### 3. Run Sample Script
Then user can go to example folder and run example_scripts.sh to check if everything goes well.  
```sh
cd example/  
bash example_scripts.sh
```
### 4. Make New Design

#### 1. Prepare Input Structures
There are several ways to obtain an antibody 3D structure, such as searching on **PDB** or modeling via **AlphaFold3**, among others.  
**AntiBMPNN** requires only the variable regions of the heavy or light chain in **PDB** file format as input.  
To proceed, create a new directory at `~/AntiBMPNN/input` and copy the PDB file into that folder.  

#### 2. Modification of Example Script  
There are some necessary variables that **must** be changed, make sure to modify them before running:

- pdb_file_path  
- CHAINS_TO_DESIGN  
- DESIGN_ONLY_POSITIONS  
- THEME  

You can also try to modify the following parameters to meet different design purposes. Please find and modify the corresponding variable values ​​in the script:  
```
    --model_name 
    --num_seq_per_target 
    --sampling_temp 
    --batch_size 
    --backbone_noise 
```
#### 3.In Silico Design Output  
The output should appear in the `~/AntiBMPNN/example` folder, where the subfolder seqs contains the output raw sequence **fasta** file and the **csv** file contains the parsed and clustered sequence.

#### 4.Validation
Users can apply **AlphaFold3** to model the designed sequence. By integrating **frequency** information, they can determine which sequence to proceed with for experimental validation.

***
   
## AntiBMPNN Performance

AntiBMPNN has a better sequence recovery rate.
<p align="center">
<img width="750" src="https://github.com/zeysun/AntiBMPNN/blob/main/figures/AntiBMPNN_3.jpg">
</p>
## Example output
Detailed output file can be found after running the `example_scripts.sh`, along with the sequence files.

| Positions(94-110) | Recovery | Frequency | PD1_volume | PD2_hydro | PD3_charge | Changes                                                                                  |
|--------------------|----------|-----------|------------|-----------|------------|------------------------------------------------------------------------------------------|
| ARSKSTYLSRDSSGYDY | 0.67     | 6495      | -0.4271    | 0.3253    | 0.9881     | [('I', 101, 'L'), ('Y', 103, 'R'), ('N', 104, 'D'), ('N', 106, 'S')]                   |
| ARSKSTYLSYNSSGYDY | 0.83     | 5722      | -0.3929    | 0.2118    | 0.9858     | [('I', 101, 'L'), ('N', 106, 'S')]                                                       |
| ARSKSTYLSYDSSGYDY | 0.75     | 5055      | -0.4071    | 0.2376    | -0.014     | [('I', 101, 'L'), ('N', 104, 'D'), ('N', 106, 'S')]                                      |

* `Positions(start-end)` - Amino acid position range of your design.
* `Recovery` - The sequence recovery rate of designed sequence.
* `Frequency` - The number of times the sequence was designed。
* `PD1_volume` - The physical descriptors of residue volume.
* `PD2_hydro` - The physical descriptors of hydrophilicity.
* `PD3_charge` - The charge properties at pH 7.4.
* `Changes` - Summary of the difference between input and output sequences.

## Model Weigts and Training Sets<br>

Link for AntiBMPNN model weights: https://zenodo.org/records/13387792/files/model_weights.zip<br>
Link for AntiBMPNN training sets: https://zenodo.org/records/13387792/files/training_set.zip

Training performance:
<p align="center">
<img width="650" src="https://github.com/zeysun/AntiBMPNN/blob/main/figures/AntiBMPNN_2.jpg">
</p>
