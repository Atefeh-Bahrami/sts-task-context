% Run the complete analysis pipeline for a given monkey and brain region.
%
% -------------------------------------------------------------------------
% REQUIRED DATA ORGANIZATION AND USAGE:
%
% After downloading the raw neural data, users must preserve the original
% monkey- and region-specific folder structure. The downloaded data should
% then be unzipped and placed inside the Exp2 directory as follows:
%
%   Exp2/
%       └── M1/
%            ├── mSTS/
%            │     ├── session1.mat
%            │     ├── session2.mat
%            │     └── ...
%            └── aSTS/
%                  ├── session1.mat
%                  ├── session2.mat
%                  └── ...
%
% The input_folder argument should point to the directory corresponding
% to a specific monkey and region (e.g., './Exp2/M1/mSTS').
%
% Example:
%   results = run_full_pipeline('./Exp2/M1/mSTS', 'M1' , 'mSTS');
%
% This pipeline must be run separately for each region (e.g., mSTS and aSTS)
% before performing any cross-region comparisons.
% -------------------------------------------------------------------------

% This function:
%   1) Builds session-level neural data structures from raw input files
%   2) Performs population-level analyses comparing task conditions
%   3) Computes region-level differences once both regions
%      have been processed


function results = run_full_pipeline(input_folder, monkey_name, region)

    fprintf('Building session data structures for monkey %s, and region %s...', ...
        monkey_name, region );

    % ---------- Session-level preprocessing ----------
    % Build standardized data structures for all sessions in the input folder

    build_all_data_structures(input_folder, monkey_name, region);

    % Path to the folder containing session-level data structures

    data_struct_folder = fullfile(input_folder, 'All_Data_Structures');

    % ---------- Population-level analyses ----------
    % Aggregate session data and compute population statistics, PSTHs,
    % dissimilarity measures, ranking analyses, and decoding-ready datasets
    
    results = make_results(data_struct_folder, monkey_name, region);

    % ---------- Cross-region comparison ----------
    % To visualize differences between regions (e.g., aSTS vs mSTS),
    % this pipeline must be run separately for each region before
    % calling the region-difference analysis
    make_regions_diff(monkey_name);
end