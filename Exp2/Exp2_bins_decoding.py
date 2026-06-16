import pandas as pd
import numpy as np
import random
import os
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from multiprocessing import Pool
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def train_model_svc(X_train, X_test, y_train, y_test):
    base_svm = svm.SVC(kernel='linear', C=1)
    ovr_model = OneVsRestClassifier(base_svm)

    ovr_model.fit(X_train, y_train)
    y_pred = ovr_model.predict(X_test)

    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred)
    return results


def save_results(prefix, all_mean_accuracy_TD_TD, all_mean_accuracy_TD_PC):
    pd.DataFrame(all_mean_accuracy_TD_TD).to_csv(
        f"svm_output/{prefix}_all_mean_accuracy_TD_TD.csv",
        index=False
    )
    pd.DataFrame(all_mean_accuracy_TD_PC).to_csv(
        f"svm_output/{prefix}_all_mean_accuracy_TD_PC.csv",
        index=False
    )


def run_for_one_bin(
    bin_num,
    data_TD,
    data_PC,
    num_iterations=1000,
    num_stimuli=40,
    num_selected_units=50,
    group_size=9,
    data_name="M2_mSTS"
):
    

    """
    Run decoding analysis for one time bin.

    Parameters
    ----------
    bin_num : int
        Index of the current time bin.
    data_TD : pandas.DataFrame
        Dataset used as the training-side condition.
    data_PC : pandas.DataFrame
        Dataset used as the cross-condition test side.
    num_iterations : int
        Number of random resampling iterations.
    num_stimuli : int
        Number of stimuli.
    num_selected_units : int
        Number of units/features sampled in each iteration.
    group_size : int
        Number of trials per stimulus; also used for within-block shuffling.
    data_name : str
        Prefix used for naming saved result files.
    """

    output_dir = Path("svm_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=output_dir/"log.txt",
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        filemode="a",
        force=True,
    )
    logging.info(f"Started bin {bin_num}")

    all_mean_accuracy_TD_TD = []
    all_mean_accuracy_TD_PC = []

    for sample in range(num_iterations):
        results_TD_TD = {}
        results_TD_PC = {}

        print("iteration:", sample, "bin_num:", bin_num, flush=True)

        selected_columns = random.sample(data_TD.columns.tolist(), num_selected_units)

        TD_selected_units = data_TD[selected_columns].copy()
        PC_selected_units = data_PC[selected_columns].copy()

        # shuffeling
        num_rows, num_columns = TD_selected_units.shape

        for col in range(num_columns):
            for group_start in range(0, num_rows, group_size):
                group_end = min(group_start + group_size, num_rows)
                group_indices = np.arange(group_start, group_end)
                np.random.shuffle(group_indices)

                TD_selected_units.iloc[group_start:group_end, col] = (
                    TD_selected_units.iloc[group_indices, col].values
                )
                PC_selected_units.iloc[group_start:group_end, col] = (
                    PC_selected_units.iloc[group_indices, col].values
                )

        # Add labels to the dataset
        TD_selected_units['label'] = 0
        PC_selected_units['label'] = 0
        rep = 9

        for i in range(num_stimuli):
            start_index = i * rep
            end_index = start_index + rep - 1
            TD_selected_units.loc[start_index:end_index, 'label'] = i
            PC_selected_units.loc[start_index:end_index, 'label'] = i

        # Apply a cross-validation approach 
        set_index = {0, 1, 2}
        map_list = [None] * (max(set_index) * 3 + 3)

        for i in set_index:
            train_index = set_index - {i}
            test_index = {i}

            for idx in test_index:
                map_list[idx * 3] = False
                map_list[idx * 3 + 1] = False
                map_list[idx * 3 + 2] = False

            for idx in train_index:
                map_list[idx * 3] = True
                map_list[idx * 3 + 1] = True
                map_list[idx * 3 + 2] = True

            all_map_list = map_list * num_stimuli

            train_set_TD = TD_selected_units[(all_map_list)]
            test_set_TD = TD_selected_units[np.logical_not(all_map_list)]

            train_set_PC = PC_selected_units[(all_map_list)]
            test_set_PC = PC_selected_units[np.logical_not(all_map_list)]

            X_train_TD = train_set_TD.iloc[:, :-1]
            y_train_TD = train_set_TD['label']

            X_test_TD = test_set_TD.iloc[:, :-1]
            y_test_TD = test_set_TD['label']

            X_train_PC = train_set_PC.iloc[:, :-1]
            y_train_PC = train_set_PC['label']

            X_test_PC = test_set_PC.iloc[:, :-1]
            y_test_PC = test_set_PC['label']

            # Standardize the data
            scaler_TD = StandardScaler()
            X_train_TD_scaled = scaler_TD.fit_transform(X_train_TD)
            X_test_TD_scaled = scaler_TD.transform(X_test_TD)
            X_test_PC_scaled = scaler_TD.transform(X_test_PC)

            # Train the classifier on TD training data and evaluate it on TD test data
            results_TD_TD[i] = train_model_svc(
                X_train_TD_scaled,
                X_test_TD_scaled,
                y_train_TD,
                y_test_TD
            )
            # Train the classifier on TD training data and evaluate it on PC test data
            results_TD_PC[i] = train_model_svc(
                X_train_TD_scaled,
                X_test_PC_scaled,
                y_train_TD,
                y_test_PC
            )

        accuracy_TD_TD = []
        accuracy_TD_PC = []

        for j in range(len(results_TD_TD)):
            accuracy_TD_TD.append(results_TD_TD[j]['accuracy'])
            accuracy_TD_PC.append(results_TD_PC[j]['accuracy'])

        mean_accuracy_TD_TD = np.mean(accuracy_TD_TD)
        mean_accuracy_TD_PC = np.mean(accuracy_TD_PC)

        all_mean_accuracy_TD_TD.append(mean_accuracy_TD_TD)
        all_mean_accuracy_TD_PC.append(mean_accuracy_TD_PC)

    # Uncomment these two lines to save results for each bin separately.
    #prefix = f"{data_name}_bin_{bin_num}"
    #save_results(prefix, all_mean_accuracy_TD_TD, all_mean_accuracy_TD_PC)


    logging.info(f"Finished bin {bin_num}")

    return {
        "bin_num": bin_num,
        "all_mean_accuracy_TD_TD": all_mean_accuracy_TD_TD,
        "all_mean_accuracy_TD_PC": all_mean_accuracy_TD_PC,
    }



def plot_binned_decoding_results(
    pc_pc_csv,
    pc_td_csv,
    td_pc_csv,
    td_td_csv,
    output_pdf="svm_bins_decoding.pdf",
    output_png="svm_bins_decoding.png",
    n_bins=18,
    data_name = 'M2_mSTS'
):
    Libre_MSB_PC_PC = pd.read_csv(pc_pc_csv)
    Libre_MSB_PC_TD = pd.read_csv(pc_td_csv)
    Libre_MSB_TD_PC = pd.read_csv(td_pc_csv)
    Libre_MSB_TD_TD = pd.read_csv(td_td_csv)

    means_main_main = [np.mean(Libre_MSB_PC_PC[str(i)]) for i in range(n_bins)]
    means_control_control = [np.mean(Libre_MSB_TD_TD[str(i)]) for i in range(n_bins)]
    means_control_main = [np.mean(Libre_MSB_TD_PC[str(i)]) for i in range(n_bins)]
    means_main_control = [np.mean(Libre_MSB_PC_TD[str(i)]) for i in range(n_bins)]

    x = [xi * 20 - 10 for xi in range(1, n_bins + 1)]

    # MATLAB-like colors
    control_dark = (0.00, 0.30, 0.60)
    main_dark = (0.80, 0.40, 0.00)
    control_light = (0.45, 0.65, 0.85)
    main_light = (0.98, 0.70, 0.45)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 18,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # within-condition
    ax.plot(x, means_control_control, color=control_dark, linewidth=2.2, label="TD_TD")
    ax.plot(x, means_main_main, color=main_dark, linewidth=2.2, label="PC_PC")

    # cross-condition
    ax.plot(x, means_control_main, color=control_light, linewidth=2.0, label="TD_PC")
    ax.plot(x, means_main_control, color=main_light, linewidth=2.0, label="PC_TD")

    ax.axvline(0, color="0.35", linewidth=1.2)
    ax.axvline(300, color="0.35", linewidth=1.2)

    ax.set_xlabel("Time (ms)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=16, fontweight="bold")

    ax.tick_params(direction="out", length=4, width=1, labelsize=18)
    ax.set_xticks([0, 300])
    ax.set_xticklabels(["0", "300"])
    ax.set_ylim(0.02, 0.11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"Bins decoding in {data_name}", fontsize=16, fontweight="bold")


    leg = ax.legend(frameon=False, fontsize=16, loc="upper right", ncols=1)

   

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")

    plt.show()


def main(data_name, num_iterations):

    
    exp_folder = Path("./svm_data")  # Folder containing the input .mat files for the decoding analysis
    
    td_path = exp_folder / f"{data_name}_all_data_res_window_bins_18_TD.mat"
    pc_path = exp_folder / f"{data_name}_all_data_res_window_bins_18_PC.mat"

    data_TD_bins_source = loadmat(td_path)
    data_PC_bins_source = loadmat(pc_path)

    data_TD_bins = data_TD_bins_source['all_data_res_window_bins_control']
    data_PC_bins = data_PC_bins_source['all_data_res_window_bins_main']

    n_workers = min(13, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1)

    print(
        
        "\nRun 1: train on TD, test on TD and PC",
      
        flush=True
    )

    # =========================================================
    # Run 1: train on TD, test on TD and PC
    # =========================================================
    datasets_args_td_pc = []
    for bin_num in range(data_TD_bins.shape[1]):
        data_TD = pd.DataFrame(data_TD_bins[:, bin_num, :])
        data_PC = pd.DataFrame(data_PC_bins[:, bin_num, :])

        datasets_args_td_pc.append(
    (
        bin_num,                 # current time bin index
        data_TD,                 # training-side dataset (TD)
        data_PC,                 # cross-condition test dataset (PC)
        num_iterations,                      # num_iterations
        40,                      # num_stimuli
        50,                      # num_selected_units
        9,                       # group_size / repetitions per stimulus
        data_name,               # output prefix 
    )
)

    with Pool(processes=n_workers) as pool:
        results_td_pc = pool.starmap(run_for_one_bin, datasets_args_td_pc)

    all_bins_mean_accuracy_TD_TD = {}
    all_bins_mean_accuracy_TD_PC = {}

    for result in results_td_pc:
        bin_num = result["bin_num"]
        all_bins_mean_accuracy_TD_TD[bin_num] = result["all_mean_accuracy_TD_TD"]
        all_bins_mean_accuracy_TD_PC[bin_num] = result["all_mean_accuracy_TD_PC"]

    df_td_td = pd.DataFrame(dict(sorted(all_bins_mean_accuracy_TD_TD.items())))
    df_td_pc = pd.DataFrame(dict(sorted(all_bins_mean_accuracy_TD_PC.items())))

    
    df_td_td.to_csv(f"svm_output/{data_name}_all_bins_mean_accuracy_TD_TD.csv", index=False)
    df_td_pc.to_csv(f"svm_output/{data_name}_all_bins_mean_accuracy_TD_PC.csv", index=False)

    


    print("Saved TD->TD and TD->PC CSV files successfully.")

    print(
        
        "\nRun 2: train on PC, test on PC and TD",
      
        flush=True
    )

    # =========================================================
    # Run 2: train on PC, test on PC and TD
    # =========================================================
    datasets_args_pc_td = []
    for bin_num in range(data_TD_bins.shape[1]):
        data_TD = pd.DataFrame(data_TD_bins[:, bin_num, :])
        data_PC = pd.DataFrame(data_PC_bins[:, bin_num, :])

        datasets_args_pc_td.append(
        (
        bin_num,                 # current time bin index
        data_PC,                 # training-side dataset (PC)
        data_TD,                 # cross-condition test dataset (TD)
        10,                      # num_iterations
        40,                      # num_stimuli
        50,                      # num_selected_units
        9,                       # group_size / repetitions per stimulus
        data_name,               # output prefix
        )
    )

    with Pool(processes=n_workers) as pool:
        results_pc_td = pool.starmap(run_for_one_bin, datasets_args_pc_td)

    all_bins_mean_accuracy_PC_PC = {}
    all_bins_mean_accuracy_PC_TD = {}

    for result in results_pc_td:
        bin_num = result["bin_num"]
        all_bins_mean_accuracy_PC_PC[bin_num] = result["all_mean_accuracy_TD_TD"]
        all_bins_mean_accuracy_PC_TD[bin_num] = result["all_mean_accuracy_TD_PC"]

    df_pc_pc = pd.DataFrame(dict(sorted(all_bins_mean_accuracy_PC_PC.items())))
    df_pc_td = pd.DataFrame(dict(sorted(all_bins_mean_accuracy_PC_TD.items())))

    df_pc_pc.to_csv(f"svm_output/{data_name}_all_bins_mean_accuracy_PC_PC.csv", index=False)
    df_pc_td.to_csv(f"svm_output/{data_name}_all_bins_mean_accuracy_PC_TD.csv", index=False)

    print("Saved PC->PC and PC->TD CSV files successfully.")

    plot_binned_decoding_results(
    pc_pc_csv=f"svm_output/{data_name}_all_bins_mean_accuracy_PC_PC.csv",
    pc_td_csv=f"svm_output/{data_name}_all_bins_mean_accuracy_PC_TD.csv",
    td_pc_csv=f"svm_output/{data_name}_all_bins_mean_accuracy_TD_PC.csv",
    td_td_csv=f"svm_output/{data_name}_all_bins_mean_accuracy_TD_TD.csv",
    output_pdf=f"svm_output/{data_name}_svm_bins_decoding.pdf",
    output_png=f"svm_output/{data_name}_svm_bins_decoding.png",
    n_bins=18,
    data_name = data_name,
)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run binned decoding analysis.")

    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help='Filename prefix, e.g. "M1_mSTS", "M1_aSTS", "M2_mSTS", or "M2_aSTS"'
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of iterations per bin"
    )

    args = parser.parse_args()
    print("Arguments parsed:", args, flush=True)

    main(
        data_name=args.data_name,
        num_iterations=args.num_iterations
    )