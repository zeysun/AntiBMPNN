# AntiBMPNN

This is the repo of AntiBMPNN project, for antibody sequence design.

<p align="center">
<img width="550" src="https://github.com/zeysun/AntiBMPNN/blob/main/figures/AntiBMPNN_1.jpg">
</p>

## Install and set up python environment

1.Make sure you have installed **conda** or **miniconda** in a **Linux** system(or **WSL**). <br>
2. `git clone https://github.com/zeysun/AntiBMPNN`  --Clone this repo to your local computer.<br>
3. `cd AntiBMPNN && bash Initialize.sh`  -- This scriot will automatically build a python environment via conda and download packages and model weights.<br>
4. Then user can go to example folder and run example_scripts.sh to check if everything goes well.<br>
  `cd example/`<br>
  `bash example_scripts.sh`<br>
5. Create a new folder in the **"input"** directory and place your antibody PDB file there. Update the corresponding variables in the `example_scripts.sh` file, and then proceed with your design.<br>
   
## AntiBMPNN performance

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

## Model weigts and training sets<br>

Link for AntiBMPNN model weights: https://zenodo.org/records/13387792/files/model_weights.zip<br>
Link for AntiBMPNN training sets: https://zenodo.org/records/13387792/files/training_set.zip

Training performance:
<p align="center">
<img width="650" src="https://github.com/zeysun/AntiBMPNN/blob/main/figures/AntiBMPNN_2.jpg">
</p>
