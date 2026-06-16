
% Compare normalized task-related response differences (PC − TD) between
% aSTS and mSTS using 10 ms binned responses and bootstrap confidence
% intervals to assess regional differences in task-effect onset latenc

function make_regions_diff( monkey_name)

bin_size = 10;
out_folder = 'diff_regions_data';
regions = {'aSTS','mSTS'};

[all_ready, missing_files] = check_region_diff_files(monkey_name, regions, bin_size, out_folder);

    if ~all_ready
        disp('Both region files are not available yet.');
        disp('Missing files:')
        disp(missing_files)
    
    else
    

    aSTS_all = load_and_normalize_region_diff(out_folder, sprintf('%s_aSTS_all_diff_original_10.mat', monkey_name), ...
    'diff_original_bins_mean', sprintf('%s_aSTS_all_ci_diff_bins_10.mat', monkey_name), 'ci_diff_bins', 1:70, 30:60);
    
    mSTS_all = load_and_normalize_region_diff(out_folder, sprintf('%s_mSTS_all_diff_original_10.mat', monkey_name), ...
    'diff_original_bins_mean', sprintf('%s_mSTS_all_ci_diff_bins_10.mat', monkey_name), 'ci_diff_bins', 1:70, 30:60);
    
    
    fig_diff_all = plot_normalized_region_diff(aSTS_all, mSTS_all);
    
    aSTS_high = load_and_normalize_region_diff(out_folder, sprintf('%s_aSTS_high_diff_original_10.mat', monkey_name), ...
    'diff_original_bins_mean', sprintf('%s_aSTS_high_ci_diff_bins_10.mat', monkey_name), 'ci_diff_bins', 1:70, 30:60);
    
    mSTS_high = load_and_normalize_region_diff(out_folder, sprintf('%s_mSTS_high_diff_original_10.mat', monkey_name), ...
    'diff_original_bins_mean', sprintf('%s_mSTS_high_ci_diff_bins_10.mat', monkey_name), 'ci_diff_bins', 1:70, 30:60);
    
    
    fig_diff_all = plot_normalized_region_diff(aSTS_high, mSTS_high);
    
    end

end


%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load bootstrapped task-difference time courses for a region and
% max-normalize the mean response difference to enable onset comparison

function diff_data = load_and_normalize_region_diff(out_folder, original_file, original_var, ci_file, ci_var, cols, norm_range)

    S1 = load(fullfile(out_folder, original_file));
    original = S1.(original_var)(:, cols);

    S2 = load(fullfile(out_folder, ci_file));
    ci = S2.(ci_var)(:, cols);

    norm_factor = max(original(1, norm_range));

    diff_data.original = original;
    diff_data.ci = ci;
    diff_data.original_norm = original / norm_factor;
    diff_data.ci_norm = ci / norm_factor;
end

function fig = plot_normalized_region_diff(ASB, MSB)

    msb_color = [0.165 0.620 0.596];
    asb_color = [0.769 0.306 0.322];

    fig = figure('Color','w');
    hold on
    ax = gca;

    num_bins = size(ASB.original_norm, 2);

    fill([1:num_bins, fliplr(1:num_bins)], ...
         [ASB.ci_norm(2,:), fliplr(ASB.ci_norm(1,:))], ...
         asb_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    fill([1:num_bins, fliplr(1:num_bins)], ...
         [MSB.ci_norm(2,:), fliplr(MSB.ci_norm(1,:))], ...
         msb_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    plot(ASB.original_norm, 'LineWidth', 2.2, 'Color', asb_color, 'DisplayName', 'aSTS');
    plot(MSB.original_norm, 'LineWidth', 2.2, 'Color', msb_color, 'DisplayName', 'mSTS');

    h3 = yline(0, 'k', 'LineWidth', 2);
    h1 = xline(30, '-', 'LineWidth', 1, 'Color', 'black');
    h2 = xline(60, '-', 'LineWidth', 1, 'Color', 'black');
    h1.HandleVisibility = 'off';
    h2.HandleVisibility = 'off';
    h3.HandleVisibility = 'off';

    xlabel('Time (ms)');
    ylabel('Normalized difference');

    set(ax, ...
        'FontName', 'Arial', ...
        'FontSize', 20, ...
        'LineWidth', 1.0, ...
        'TickDir', 'out', ...
        'Box', 'off', ...
        'XLim', [0 70], ...
        'XTick', [30 60], ...
        'XTickLabel', {'0','300'}, ...
        'YLim', [-1.2 1.6]);

    hold off
end


% Check whether all required region-level task-difference files exist
% before performing cross-region comparison

function [all_ready, missing_files] = check_region_diff_files(monkey_name, region_names, bin_size, out_folder)

    if nargin < 4
        out_folder = 'diff_regions_data';
    end

    missing_files = {};
    all_ready = true;

    for r = 1:numel(region_names)
        region = region_names{r};

        files_to_check = {
            sprintf('%s_%s_all_diff_original_%d.mat', monkey_name, region, bin_size)
            sprintf('%s_%s_all_ci_diff_bins_%d.mat', monkey_name, region, bin_size)
            sprintf('%s_%s_high_diff_original_%d.mat', monkey_name, region, bin_size)
            sprintf('%s_%s_high_ci_diff_bins_%d.mat', monkey_name, region, bin_size)
        };

        for k = 1:numel(files_to_check)
            full_path = fullfile(out_folder, files_to_check{k});
            if ~isfile(full_path)
                all_ready = false;
                missing_files{end+1,1} = full_path;
            end
        end
    end
end