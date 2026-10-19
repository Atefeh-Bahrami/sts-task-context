# Neural Data Analysis Pipeline

This repository contains MATLAB pipelines for population-level analysis of neural recordings across two experiments presented in the paper:

**"Task Context Modulates the Selectivity of Macaque Superior Temporal Sulcus Neurons for Partially Occluded Bodies"**.
---
## Requirements

- MATLAB R2024b (Update 2)
- Statistics and Machine Learning Toolbox (Version 24.2)

Both Experiment 1 and Experiment 2 use the same MATLAB dependencies.

## Code Clone
Clone this repository:

```bash
git clone https://github.com/Atefeh-Bahrami/sts-task-context.git
```

The code is organized into two directories:

- `Exp1/`: MATLAB scripts and functions for Experiment 1
- `Exp2/`: MATLAB scripts and functions for Experiment 2

## MATLAB Setup

Open MATLAB and navigate to the corresponding experiment directory (`Exp1` or `Exp2`).

For Experiment 1:

```matlab
cd('your_local_path/sts-task-context/Exp1')
```

For Experiment 2:

```matlab
cd('your_local_path/sts-task-context/Exp2')
```

Ensure that the required MATLAB functions are accessible from the current directory or MATLAB search path before running the analysis.



## Data Download

Download the dataset from the link below and extract it locally before running the analysis:

https://doi.org/10.5281/zenodo.20832778
---
Download all three files from the Zenodo repository:

Exp1-Data-Neuro.zip — Neural recording data for Experiment 1.

Exp2-Data-Neuro.zip — Neural recording data for Experiment 2.

ReadMe.txt — Additional information about the dataset.

Extract the ZIP archives and place the data in the corresponding experiment directories within the cloned repository:

Experiment 1: sts-task-context/Exp1/Data-Neuro/

Experiment 2: sts-task-context/Exp2/Data-Neuro/

Ensure that the extracted data follow the directory structure described below before running the MATLAB pipelines.

## Data Organization

Ensure that the data are organized as follows:
```
Exp1/
└── Data-Neuro/
    ├── M1/
    │   ├── session1.mat
    │   ├── session2.mat
    │   └── ...
    └── M2/
        ├── session1.mat
        ├── session2.mat
        └── ...

Exp2/
└── Data-Neuro/
    ├── M1/
    │   ├── aSTS/
    │   │   ├── session1.mat
    │   │   └── ...
    │   └── mSTS/
    │       ├── session1.mat
    │       └── ...
    └── M2/
        ├── aSTS/
        │   ├── session1.mat
        │   └── ...
        └── mSTS/
            ├── session1.mat
            └── ...
```


## Running Experiment 1 (MATLAB pipeline)
Before running the pipeline, ensure that MATLAB's current working directory is set to Exp1.
Run the main pipeline separately for each monkey by specifying the corresponding data folder and monkey identifier (M1 or M2).
For example: 
```matlab
results = run_full_pipeline('your_local_path/sts-task-context/Exp1/Data-Neuro/M1', 'M1');
```
After successfully running the MATLAB pipeline, a folder named All_Data_Structures is automatically created within the corresponding monkey's data directory to store the generated data structures.

For Experiment 1:

- Exp1/Data-Neuro/M1/All_Data_Structures/
- Exp1/Data-Neuro/M2/All_Data_Structures/

## Pipeline overview
```markdown
- Aggregates session-level neural data
- Computes stimulus dissimilarity matrices
- Performs statistical comparisons between task conditions
- Generates population PSTHs (raw and normalized)
- Exports data for decoding analyses
```

## Running Experiment 2 (MATLAB pipeline)
Before running the pipeline, ensure that MATLAB's current working directory is set to Exp2.
Run the main pipeline separately for each monkey and recording region by specifying the corresponding data folder, monkey identifier (M1 or M2), and region name (aSTS or mSTS).
For example: 

```matlab
results = run_full_pipeline('your_local_path/sts-task-context/Exp2/Data-Neuro/M1/mSTS', 'M1', 'mSTS');
```
After successfully running the MATLAB pipeline, a folder named All_Data_Structures is automatically created within the corresponding monkey's data directory to store the generated data structures.

For Experiment 2:

- Exp2/Data-Neuro/M1/aSTS/All_Data_Structures/
- Exp2/Data-Neuro/M1/mSTS/All_Data_Structures/
- Exp2/Data-Neuro/M2/aSTS/All_Data_Structures/
- Exp2/Data-Neuro/M2/mSTS/All_Data_Structures/


## Pipeline overview
```markdown
- Aggregates session-level neural data
- Computes stimulus dissimilarity matrices
- Performs statistical comparisons between task conditions
- Generates population PSTHs (raw and normalized)
- Exports data for decoding analyses
- Performs cross-validated stimulus-ranking analysis  
- Computes region-level response differences  
```

## Output

Running the MATLAB pipelines generates:

- Stimulus dissimilarity matrices
- Population PSTHs (raw and normalized)
- Cross-validated stimulus-ranking plots (Experiment 2)
- Region-level response difference plots (Experiment 2)
- Processed neural data for downstream analyses (e.g., SVM decoding)

## Reproducibility Notes

The number of units included in the analyses may vary slightly between runs and may differ from the numbers reported in the published figures. This variability arises from the stochastic nature of the reliability estimation procedure, which can result in small differences in the number of units meeting the reliability criterion.

These minor variations do not affect the main findings or conclusions of the study.

## Citation

Nejad GG, Bahrami A, Bognár A, Mukovskiy A, Giese M, Vogels R (2026). Task Context Modulates the Selectivity of Macaque Superior Temporal Sulcus Neurons for Partially Occluded Bodies. Journal of Neuroscience, 46(38), e0661262026.

DOI: https://doi.org/10.1523/JNEUROSCI.0661-26.2026
