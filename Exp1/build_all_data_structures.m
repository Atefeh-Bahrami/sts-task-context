
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build_all_data_structures.m
%
% This script processes raw session-level neural data files and builds
% standardized data structures used for all downstream population and
% decoding analyses.
%
% For each session, the pipeline:
%   1) Extracts basic session metadata and stimulus information
%   2) Computes baseline and stimulus-evoked firing rates
%   3) Identifies responsive and selective units using statistical tests
%   4) Applies z-score and reliability criteria for unit selection
%   5) Computes mean responses across stimuli and repetitions
%   6) Identifies preferred-stimulus responses per unit
%   7) Prepares trial-aligned responses for SVM decoding
%   8) Saves a session-level data structure for later analysis
%
% This function should be run before any population-level analyses
% (e.g., make_results).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Session-level data structure construction %%%%%%%%%%%%

function build_all_data_structures(input_folder, monkey)

    files = dir(fullfile(input_folder, '*.mat'));
    
    for k = 1:length(files)
        filename = files(k).name;
        full_input_path = fullfile(files(k).folder, filename);
    
        [~, name, ~] = fileparts(filename);
        parts = split(name, '_');
        task = string(parts{end});  
    
        S = load(full_input_path);
        responsesAcrossTests = S.responsesAcrossTests;
    
        
        info = extract_basic_session_info(responsesAcrossTests, monkey, task);
        baseline_window = 150:349;   % 200 ms pre-stimulus baseline window
        stimulus_window = 350:549;   % 200 ms post-stimulus response window
        
        % Compute baseline and stimulus-evoked firing rates for all trials and channels
        [baseline_firing, response_firing] = compute_baseline_response_firing(info , baseline_window , stimulus_window);
        
        %%%%%%%%%%%%%% Statistical tests for identifying responsive and selective units %%%%%%%%%%%%%
        %Split-plot ANOVA and Kruskal–Wallis nonparametric test 
        results = find_significant_channels(baseline_firing, response_firing, info.stimuli_num, info.repetition, info.ch_num);
        significant_channels = results.significant_channels;
        
        % Compute z-scores of stimulus responses relative to baseline activity
        channels_above_z4 = find_channels_above_zscore( significant_channels, baseline_firing, info.data, info.stimuli_num, info.repetition);
        
        % Compute unit reliability using split-half correlations with Spearman–Brown correction
        reliability_mean_corrected_corr = compute_unit_reliability(response_firing, info.all_channels, info.stimuli_num, info.repetition);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Mean responses across repetitions (trials) and stimuli %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [mean_response, mean_baseline, mean_response_all_stimuli] = compute_mean_responses_across_stimuli(response_firing, baseline_firing, info.stimuli_num, info.repetition);

        % Identify the preferred stimulus for each channel and compute the
        % corresponding mean responses, as well as mean responses across all stimuli
        
        [high_stimuli_response, data_high_stimuli, data_all_stimuli] =  compute_all_stimuli_and_best_stimulus_responses(mean_response, mean_baseline, info.data, info.repetition);
        
        %%%%%%%%%%%%%%%%%%%%%%%%% Prepare neural data for SVM decoding %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        response_firing_medium_stimuli = get_medium_stimuli_response(response_firing, info.stimuli_num, info.repetition, monkey);
        
        
        %%%%%%%%%%%%%%%%%%%%%%% Build and save session-level data structure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        this_file = fileparts(mfilename('fullpath'));
        project_root = fileparts(this_file);
        folder_name = fullfile(project_root, 'Exp1', 'Data-Neuro', char(monkey), 'All_Data_Structures');
        
        
         % Create the folder if it doesn't exist
         if ~exist(folder_name, 'dir')
            mkdir(folder_name);
         end
        
         data_struct.data_name = info.data_name;
         data_struct.channel_names = info.ch_names;
         data_struct.channels_above_z4 = channels_above_z4;
         data_struct.reliability = reliability_mean_corrected_corr(info.all_channels);
         data_struct.mean_response_all_stimuli = mean_response_all_stimuli;
         data_struct.mean_response = mean_response;
         data_struct.stimlist=info.unique_stimuli;
         data_struct.high_stimuli_response = high_stimuli_response;
         data_struct.all_stimuli_response = data_all_stimuli;
         data_struct.data_high_stimuli =data_high_stimuli;
         data_struct.response_firing_medium_stimuli=response_firing_medium_stimuli;
         
         
         % Save the structure
         filename_all = sprintf('%s.mat', info.data_name);
         full_path = fullfile(folder_name, filename_all);
         save(full_path, 'data_struct');

    end
end


 


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% preprocessing functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % Extract basic session metadata and experimental parameters
 % from the raw response data, including stimulus information,
 % repetition counts, channel identities, task type, and session name

 function info = extract_basic_session_info(responsesAcrossTests, monkey, task)

    info.stimuli_num = size(responsesAcrossTests(1).mpc , 1);
    info.repetition = responsesAcrossTests(1).numTrial * size(responsesAcrossTests, 2);
    info.data = responsesAcrossTests(1).rasterAll;
    info.monkey_name = monkey;

    info.trials = info.stimuli_num * info.repetition;
    info.ch_num = size(responsesAcrossTests(1).raster, 3);
    info.ch_names = responsesAcrossTests(1).Chan_name;
    info.task = "_" + task;

    info.data_name = regexprep(responsesAcrossTests(1).filename, '_\d+_[AB]', '');
    info.data_name = string(info.data_name) + info.task;

    info.stimuli_list = responsesAcrossTests(1).Stimlist;
    info.unique_stimuli = unique(info.stimuli_list);
    info.all_channels = 1:length(info.ch_names);

 end


 % Compute baseline and stimulus-evoked firing rates for each trial
 % and channel by summing spike counts within predefined time windows
 function [baseline_firing, response_firing] = compute_baseline_response_firing(info , baseline_window , stimulus_window)

    baseline_firing = zeros(info.trials, info.ch_num);
    response_firing = zeros(info.trials, info.ch_num);

    for stim = 1:info.trials
        for ch = 1:info.ch_num
            baseline_data = info.data(stim, baseline_window, ch);
            stimulus_data = info.data(stim, stimulus_window, ch);

            baseline_firing(stim, ch) = sum(baseline_data) * 5; % ms to s;
            response_firing(stim, ch) = sum(stimulus_data) * 5;
        end
    end
 end

 % Identify responsive and stimulus-selective channels using
 % split-plot ANOVA and Kruskal–Wallis tests, and return channels that
 % are significant in both tests
 function results = find_significant_channels(baseline_firing, response_firing, stimuli_num, repetition, ch_num)

    significant_channels_anova = {};
    significant_channels_kruskal_test = {};

    for ch = 1:ch_num

        data_matrix_anova = zeros(stimuli_num * repetition, 3);

        for stim = 1:stimuli_num
            for rep = 1:repetition
                trial_index = (stim - 1) * repetition + rep;
                condition_code = stim;

                data_matrix_anova(trial_index, 1) = condition_code;
                data_matrix_anova(trial_index, 2) = baseline_firing(trial_index, ch);
                data_matrix_anova(trial_index, 3) = response_firing(trial_index, ch);
            end
        end

        structStat = splitplotANOVA(data_matrix_anova);
        kruskal_p = kruskalwallis(data_matrix_anova(:,3), data_matrix_anova(:,1), 'off');

        if structStat.significant == 1
            significant_channels_anova{end+1} = sprintf('Channel %d', ch);
        end

        if kruskal_p < 0.05
            significant_channels_kruskal_test{end+1} = sprintf('Channel %d', ch);
        end
    end

    results.significant_channels_anova = significant_channels_anova;
    results.significant_channels_kruskal_test = significant_channels_kruskal_test;
    results.significant_channels = intersect(significant_channels_anova, significant_channels_kruskal_test);
 end


 %Further filter statistically significant channels by testing whether
 % stimulus responses exceed baseline activity by a z-score threshold
 % (z > 4) in at least one response window
 function channels_above_z4 = find_channels_above_zscore(significant_channels, baseline_firing, data, stimuli_num, repetition)

    channels_above_z4 = [];

    for ch = 1:length(significant_channels)

        current_channel = significant_channels{ch};
        channel_number = regexp(current_channel, '\d+', 'match');
        channel_number = str2double(channel_number{1});

        [mean_baselines, std_baseline] = compute_mean_std_firing( ...
            baseline_firing, channel_number, stimuli_num, repetition);

        mean_firing_per_window = compute_mean_per_win( ...
            data, channel_number, stimuli_num, repetition, 1);

        z_score_per_win = compute_zscore( ...
            mean_firing_per_window, mean_baselines, std_baseline, stimuli_num, 1);

        if any(z_score_per_win > 4, 'all')
            channels_above_z4 = [channels_above_z4, channel_number];
        end
    end
end



% Compute baseline statistics for a single channel by:
%   1) Averaging baseline firing rates across repetitions for each stimulus
%   2) Computing the mean and standard deviation across stimuli 
function [mean_baseline, std_baseline] = compute_mean_std_firing(baseline_firing, channel_num, stimuli_num, repitation)
     
     mean_per_stimulus = zeros(1, stimuli_num);
     
     for stim = 1:stimuli_num
         start_idx = (stim - 1) * repitation + 1;
         end_idx = stim * repitation;
         stimuli_firing_rates = baseline_firing(start_idx:end_idx, channel_num)/5; % Convert spike counts to firing rates (spikes/s)
         mean_per_stimulus(stim) = mean(stimuli_firing_rates);
         %fprintf('mean_per_stimulus: %d\n', mean_per_stimulus(stim)); 
     end
     
     mean_baseline = mean(mean_per_stimulus);
     std_baseline = std(mean_per_stimulus);
         
 end
 

 
function [mean_firing_per_window] = compute_mean_per_win(data_raster, channel_num, stimuli_num, repitation, win_num)
     % Initialize parameters
     win_length = 200;              % Length of each window
     num_windows = win_num;               % Number of windows
     start_point = 350;             % Starting point for the windows
 
     response_stimuli_firing_win = zeros(stimuli_num, num_windows);  
     
     for stim = 1:stimuli_num      
         for win = 1:num_windows            
             firing_rate_sum = 0;            
             for rep = 1:repitation
                 stim_idx = (stim - 1) * repitation + rep;
                 start_idx = start_point + (win - 1) * win_length;  
                 end_idx = start_idx + win_length - 1;              
                 stimulus_data = data_raster(stim_idx, start_idx:end_idx, channel_num);
                 firing_rate = sum(stimulus_data); 
                 firing_rate_sum = firing_rate_sum + firing_rate;
             end
 
             mean_firing_rate = firing_rate_sum / repitation;
             response_stimuli_firing_win(stim, win) = mean_firing_rate;  
         end
     end
     mean_firing_per_window = response_stimuli_firing_win;  
 end
 
  
 function [z_score_per_win] = compute_zscore(mean_firing_per_window, baseline_mean, baseline_std, stimuli_num, num_windows)
     z_score_per_win = zeros(stimuli_num, num_windows); 
 
     for stim = 1:stimuli_num
         for win = 1:num_windows
             z_score_per_win(stim, win) = (mean_firing_per_window(stim, win) - baseline_mean) / baseline_std;
         end
     end
  end
 
 % Estimate unit reliability using split-half correlations across
 % trials and stimuli, with Spearman–Brown correction applied
 % Correlations are computed after shuffling trial order within stimuli
 function reliability_mean_corrected_corr = compute_unit_reliability(response_firing, all_channels, stimuli_num, repetition)

    if mod(repetition, 2) == 1
        half1 = fix(repetition / 2);
        half2 = fix(repetition / 2) + 1;
    else
        half1 = fix(repetition / 2);
        half2 = fix(repetition / 2);
    end

    response_firing_shuffled = response_firing;

    for j = 1:100

        for stim = 1:stimuli_num
            row_indices = (stim - 1) * repetition + (1:repetition);
            response_firing_shuffled(row_indices, :) = response_firing(row_indices(randperm(repetition)), :);
        end

        for i = 1:length(all_channels)
            chan = all_channels(i);

            for stim = 1:stimuli_num
                means_responses_part1(stim, chan) = mean(response_firing_shuffled((stim - 1) * repetition + 1 : stim * repetition - half2, chan));
                means_responses_part2(stim, chan) = mean(response_firing_shuffled((stim - 1) * repetition + half1 + 1 : stim * repetition, chan));
            end

            correlation_shuffled_stim(j, chan) = corr(means_responses_part1(:, chan), means_responses_part2(:, chan), 'Type', 'Pearson');
        end
    end

    n = 2;
    for i = 1:length(all_channels)
        chan = all_channels(i);

        for j = 1:size(correlation_shuffled_stim, 1)
            correlation = correlation_shuffled_stim(j, chan);
            sp_brown_corrected_correlations(j, chan) = (n * correlation) / (1 + (n - 1) * correlation);
        end
    end

    reliability_mean_corrected_corr = mean(sp_brown_corrected_correlations);
end 
  


function [mean_response, mean_baseline, mean_response_all_stimuli] = compute_mean_responses_across_stimuli(response_firing, baseline_firing, stimuli_num, repetition)

    ch_num = size(response_firing, 2);

    mean_response = zeros(stimuli_num, ch_num);
    mean_baseline = zeros(stimuli_num, ch_num);

    for stim = 1:stimuli_num
        row_indices = (stim - 1) * repetition + (1:repetition);

        mean_response(stim, :) = mean(response_firing(row_indices, :), 1);
        mean_baseline(stim, :) = mean(baseline_firing(row_indices, :), 1);
    end

    mean_response_all_stimuli = mean(mean_response, 1);
end

% Identify the preferred stimulus for each channel based on net firing rate
% and compute corresponding mean responses, as well as mean responses
% averaged across all stimuli

function [high_stimuli_response, data_high_stimuli, data_all_stimuli] =  compute_all_stimuli_and_best_stimulus_responses(mean_response, mean_baseline, data, repetition)

    net_firing_rate = mean_response - mean_baseline;

    [high_stimuli_response, row_indices_max_stim] = max(net_firing_rate, [], 1);

    ch_num = size(net_firing_rate, 2);
    time_num = size(data, 2);

    data_high_stimuli = zeros(ch_num, time_num);

    for chan = 1:ch_num
        rows = (row_indices_max_stim(chan) - 1) * repetition + 1 : row_indices_max_stim(chan) * repetition;
        data_high_stimuli(chan, :) = mean(data(rows, :, chan), 1);
    end

    data_all_stimuli = squeeze(mean(data, 1))';
end
  


% Reformat stimulus response data to equalize repetition counts across
% sessions and apply monkey-specific adjustments to handle missing stimuli
% in the medium-size condition (e.g., missing stimulus in M2) for SVM decoding

function response_firing_medium_stimuli = get_medium_stimuli_response(response_firing, stimuli_num, repetition, monkey_name)

    selected_reps = 8;

    if repetition == 8
        response_firing_model = response_firing;

    elseif repetition == 10
        ch_num = size(response_firing, 2);
        response_firing_model = zeros(stimuli_num * selected_reps, ch_num);

        for stim = 1:stimuli_num
            start_idx = (stim - 1) * repetition + 1;
            end_idx = start_idx + selected_reps - 1;
            selected_data = response_firing(start_idx:end_idx, :);

            output_start = (stim - 1) * selected_reps + 1;
            output_end = output_start + selected_reps - 1;

            response_firing_model(output_start:output_end, :) = selected_data;
        end
    else
        error('Unsupported repetition value: %d', repetition);
    end

    if monkey_name == "M2"

        temp_medium = zeros(16, size(response_firing_model, 2));
        temp_medium(1:8, :) = response_firing_model(1:8, :);
        temp_medium(9:16, :) = response_firing_model(25:32, :);

        response_firing_model2 = response_firing_model(65:952, :);
        response_firing_model_filtered = zeros(37 * selected_reps, size(response_firing_model, 2));

        for stim = 1:37
            start_idx = (stim - 1) * selected_reps * 3 + 1;
            end_idx = start_idx + selected_reps - 1;
            selected_data = response_firing_model2(start_idx:end_idx, :);

            output_start = (stim - 1) * selected_reps + 1;
            output_end = output_start + selected_reps - 1;

            response_firing_model_filtered(output_start:output_end, :) = selected_data;
        end

        response_firing_model_filtered = [zeros(16, size(response_firing_model_filtered, 2)); response_firing_model_filtered];
        response_firing_model_filtered(1:16, :) = temp_medium;

        response_firing_medium_stimuli = response_firing_model_filtered;

    elseif monkey_name == "M1"

        response_firing_model2 = response_firing_model;
        nstim = stimuli_num / 3;
        response_firing_model_filtered = zeros(nstim * selected_reps, size(response_firing_model, 2));

        for stim = 1:nstim
            start_idx = (stim - 1) * selected_reps * 3 + 1;
            end_idx = start_idx + selected_reps - 1;
            selected_data = response_firing_model2(start_idx:end_idx, :);

            output_start = (stim - 1) * selected_reps + 1;
            output_end = output_start + selected_reps - 1;

            response_firing_model_filtered(output_start:output_end, :) = selected_data;
        end

        response_firing_medium_stimuli = response_firing_model_filtered;

    else
        error('Unknown monkey name: %s', monkey_name);
    end
end