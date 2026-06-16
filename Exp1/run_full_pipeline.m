% Run the complete analysis pipeline for a given monkey.
%
% -------------------------------------------------------------------------
% REQUIRED DATA ORGANIZATION AND USAGE:
%
% After downloading the raw neural data, users must preserve the original
% monkey-specific folder structure. The downloaded data should
% then be unzipped and placed inside the Exp1 directory as follows:
%
%   Exp1/
%       └── M1/
%            ├── session1.mat
%            ├── session2.mat
%            └── ...
%       └── M2/
%            ├── session1.mat
%            ├── session2.mat
%            └── ...
%
% The input_folder argument should point to the directory corresponding
% to a specific monkey (e.g., './Exp1/M1').
% Example:
%   results = run_full_pipeline('./Exp1/Data-Neuro/M1', 'M1');
%
% -------------------------------------------------------------------------
% This function:
%   1) Builds session-level neural data structures from raw input files
%   2) Performs population-level analyses comparing task conditions

function results = run_full_pipeline(input_folder, monkey_name)

    % ---------- Session-level preprocessing ----------
    % Build standardized data structures for all sessions in the input folder

    build_all_data_structures(input_folder, monkey_name);

    % Path to the folder containing session-level data structures
    data_struct_folder = fullfile(input_folder, 'All_Data_Structures');


    % ---------- Population-level analyses ----------
    % Aggregate session data and compute population statistics, PSTHs,
    % dissimilarity measures, and decoding-ready datasets
    results = make_results(data_struct_folder, monkey_name);
end