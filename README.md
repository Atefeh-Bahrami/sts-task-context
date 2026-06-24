# Neural Data Analysis Pipeline

This repository contains MATLAB pipelines for population-level analysis of neural recordings across two experiments, together with Python scripts for decoding analyses.
---

## Code Clone

To use the code, first clone this repository:

git clone https://github.com/Atefeh-Bahrami/sts-task-context.git

Then go to the project folder:

cd sts-task-context

The code is organized into separate folders for Experiment 1 and Experiment 2:

Exp1/: scripts and notebooks for Experiment 1
Exp2/: scripts and notebooks for Experiment 2


## Data Download

Download the dataset from the link below and extract it locally before running the analysis:

https://doi.org/10.5281/zenodo.20832778
---

## Data Organization

Ensure that the data are organized as follows:
```
Exp1/
  └── Data-Neuro/
    └── MonkeyName/
        ├── session1.mat
        ├── session2.mat
        └── ...
Exp2/
└── Data-Neuro/
    └── MonkeyName/
      └── RegionName/
          ├── session1.mat
          ├── session2.mat
          └── ...
```
## Running Experiment 1 (MATLAB pipeline)

Run the main pipeline by specifying the path to the corresponding data folder:

```matlab
results = run_full_pipeline('your_local_path/Exp1/Data-Neuro/M1', 'M1');
```

## Pipeline overview
```markdown
- Aggregates session-level neural data
- Computes stimulus dissimilarity matrices
- Performs statistical comparisons between task conditions
- Generates population PSTHs (raw and normalized)
- Exports data for decoding analyses
```

## Running Experiment 2 (MATLAB pipeline)

Run the main pipeline by specifying the path to the corresponding data folder:

```matlab
results = run_full_pipeline('your_local_path/Exp2/Data-Neuro/M1/mSTS', 'M1', 'mSTS');
```

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

# Output
Running the pipeline will:
```markdown
Generate figures (dissimilarity matrices, PSTHs, ranking plots, region-level response differences plots)
Save processed data for downstream analyses (e.g., SVM decoding)
```
