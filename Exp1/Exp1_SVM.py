import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random



def train_model_svc(X_train, X_test, y_train, y_test):
    base_svm = svm.SVC(kernel='linear', C=1)
    ovr_model = OneVsRestClassifier(base_svm)

    ovr_model.fit(X_train, y_train)
    y_pred = ovr_model.predict(X_test)

    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred)
    #results["f1_score"] = f1_score(y_test, y_pred, average='weighted')
    #results["precision"] = precision_score(y_test, y_pred, average='weighted')
    #results["recall"] = recall_score(y_test, y_pred, average='weighted')

    return results


def run_decoding_analysis(
    data_TD,
    data_PC,
    data_name = "Ex1_M2",
    num_iterations=1000,
    num_stimuli=40,
    num_selected_units=50,
    group_size=8,
):
    all_mean_accuracy_TD_TD = []
    all_mean_accuracy_TD_PC = []

    for sample in range(num_iterations):
        results_TD_TD = {}
        results_TD_PC = {}

        print("iteration:", sample)

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

        # adding label to dataset
        TD_selected_units['label'] = 0
        PC_selected_units['label'] = 0

        rep = group_size
        for i in range(num_stimuli):
            start_index = i * rep
            end_index = start_index + rep - 1
            TD_selected_units.loc[start_index:end_index, 'label'] = i
            PC_selected_units.loc[start_index:end_index, 'label'] = i

        # cross validation
        set_index = {0, 1, 2, 3}
        map_list = [None] * (max(set_index) * 2 + 2)

        for i in set_index:
            train_index = set_index - {i}
            test_index = {i}

            for idx in test_index:
                map_list[idx * 2] = False
                map_list[idx * 2 + 1] = False

            for idx in train_index:
                map_list[idx * 2] = True
                map_list[idx * 2 + 1] = True

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
            scaler = StandardScaler()
            X_train_TD_scaled = scaler.fit_transform(X_train_TD)
            X_test_TD_scaled = scaler.transform(X_test_TD)
            X_test_PC_scaled = scaler.transform(X_test_PC)

            results_TD_TD[i] = train_model_svc(
                X_train_TD_scaled,
                X_test_TD_scaled,
                y_train_TD,
                y_test_TD
            )
            
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

    pd.Series(
        all_mean_accuracy_TD_PC, name="value"
    ).to_csv(f"{data_name}_all_mean_accuracy_TD_PC.csv", index=False)

    pd.Series(
        all_mean_accuracy_TD_TD, name="value"
    ).to_csv(f"{data_name}_all_mean_accuracy_TD_TD.csv", index=False)


    return {
        "all_mean_accuracy_TD_TD": all_mean_accuracy_TD_TD,
        "all_mean_accuracy_TD_PC": all_mean_accuracy_TD_PC,
    }




def run_statistical_test_analysis_label_shuffling(
    data_TD,
    data_PC,
    data_name="Ex1_M1",
    num_stat_tests=500,
    num_iterations_per_test=100,
    num_stimuli=40,
    stimuli_trials_size=8,
    num_selected_units=50,
):
    all_500_mean_accuracy_TD_TD = []
    all_500_mean_accuracy_PC_PC = []

    def make_mixed(TD_df, PC_df, block_size):
        assert TD_df.shape == PC_df.shape
        nrows, _ = TD_df.shape

        mixed_blocks = []
        for start in range(0, nrows, block_size):
            end = min(start + block_size, nrows)
            mixed_blocks.append(TD_df.iloc[start:end])  # TD block
            mixed_blocks.append(PC_df.iloc[start:end])     # PC block

        mixed = pd.concat(mixed_blocks, axis=0, ignore_index=True)
        return mixed

    for st_test in range(num_stat_tests):  # 500 iterations for statistical test
        print("Iteration_statistic:", st_test, f"/{num_stat_tests}")

        all_mean_accuracy_TD_TD = []
        all_mean_accuracy_PC_PC = []

        # merging the trials of PC and TD
        TD_PC_mix_trials = make_mixed(
            data_TD, data_PC, block_size=stimuli_trials_size
        )

        # shuffeling the trials of PC and TD
        num_rows, num_columns = TD_PC_mix_trials.shape
        group_size = 2 * stimuli_trials_size

        for col in range(num_columns):
            for group_start in range(0, num_rows, group_size):
                group_end = min(group_start + group_size, num_rows)
                group_len = group_end - group_start
                block_len = group_len // 2
                mixed_vals = TD_PC_mix_trials.iloc[group_start:group_end, col].to_numpy()

                # shuffle this mixed block
                perm = np.random.permutation(group_len)
                mixed_vals = mixed_vals[perm]

                block_idx = group_start // group_size
                orig_start = block_idx * stimuli_trials_size
                orig_end = orig_start + block_len

                # first half -> TD, second half -> PC
                data_TD.iloc[orig_start:orig_end, col] = mixed_vals[:block_len]
                data_PC.iloc[orig_start:orig_end, col] = mixed_vals[block_len:block_len + block_len]

        for sample in range(num_iterations_per_test):
            results_TD_TD = {}
            results_PC_PC = {}
            print("iteration:", sample, "of test num:", st_test)

            selected_columns = random.sample(data_TD.columns.tolist(), num_selected_units)
            TD_selected_units = data_TD.loc[:, selected_columns].copy()
            PC_selected_units = data_PC.loc[:, selected_columns].copy()

            # adding lable to dataset
            TD_selected_units['label'] = 0
            PC_selected_units['label'] = 0
            rep = 8

            for i in range(num_stimuli):
                start_index = i * rep
                end_index = start_index + rep - 1
                TD_selected_units.loc[start_index:end_index, 'label'] = i
                PC_selected_units.loc[start_index:end_index, 'label'] = i

            # cross validation
            set_index = {0, 1, 2, 3}
            map_list = [None] * (max(set_index) * 2 + 2)

            for i in set_index:

                train_index = set_index - {i}
                test_index = {i}

                for idx in test_index:
                    map_list[idx * 2] = False
                    map_list[idx * 2 + 1] = False

                for idx in train_index:
                    map_list[idx * 2] = True
                    map_list[idx * 2 + 1] = True

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
                scaler_TD = StandardScaler().fit(X_train_TD)
                X_train_TD_scaled = scaler_TD.transform(X_train_TD)
                X_test_TD_scaled = scaler_TD.transform(X_test_TD)

                result_TD = train_model_svc(
                    X_train_TD_scaled,
                    X_test_TD_scaled,
                    y_train_TD,
                    y_test_TD
                )
                results_TD_TD[i] = result_TD

                ####### PC_PC
                scaler_PC = StandardScaler().fit(X_train_PC)
                X_train_PC_scaled = scaler_PC.transform(X_train_PC)
                X_test_PC_scaled = scaler_PC.transform(X_test_PC)

                result_PC = train_model_svc(
                    X_train_PC_scaled,
                    X_test_PC_scaled,
                    y_train_PC,
                    y_test_PC
                )
                results_PC_PC[i] = result_PC

            ####### TD_TD
            accuracy_TD_TD = []
            for j in range(len(results_TD_TD)):
                accuracy_TD_TD.append(results_TD_TD[j]['accuracy'])

            mean_accuracy_TD_TD = np.mean(accuracy_TD_TD)
            all_mean_accuracy_TD_TD.append(mean_accuracy_TD_TD)

            ####### PC_PC
            accuracy_PC_PC = []
            for j in range(len(results_PC_PC)):
                accuracy_PC_PC.append(results_PC_PC[j]['accuracy'])

            mean_accuracy_PC_PC = np.mean(accuracy_PC_PC)
            all_mean_accuracy_PC_PC.append(mean_accuracy_PC_PC)

        all_500_mean_accuracy_TD_TD.append(np.mean(all_mean_accuracy_TD_TD))
        all_500_mean_accuracy_PC_PC.append(np.mean(all_mean_accuracy_PC_PC))

    pd.Series(
        all_500_mean_accuracy_PC_PC, name="value"
    ).to_csv(f"{data_name}_all_500_mean_accuracy_PC_PC.csv", index=False)

    pd.Series(
        all_500_mean_accuracy_TD_TD, name="value"
    ).to_csv(f"{data_name}_all_500_mean_accuracy_TD_TD.csv", index=False)

    return {
        "all_500_mean_accuracy_TD_TD": all_500_mean_accuracy_TD_TD,
        "all_500_mean_accuracy_PC_PC": all_500_mean_accuracy_PC_PC,
    }



def run_original_data_statistic_analysis(
    data_TD,
    data_PC,
    data_name="Ex1-M2",
    num_iterations=100,
    num_stimuli=39,
    num_selected_units=50,
    group_size=8,
):
    all_mean_accuracy_TD_TD = []  

    all_mean_accuracy_PC_PC = []
    
    for sample in range(num_iterations):
        results_TD_TD = {}
        results_PC_PC = {}
        print("iteration:", sample)

        selected_columns = random.sample(data_TD.columns.tolist(), num_selected_units)

        TD_selected_units = data_TD.loc[:, selected_columns].copy()
        PC_selected_units = data_PC.loc[:, selected_columns].copy()

        # shuffeling
        num_rows, num_columns = TD_selected_units.shape

        for col in range(num_columns):
            for group_start in range(0, num_rows, group_size):
                group_end = min(group_start + group_size, num_rows)
                group_indices = np.arange(group_start, group_end)
                np.random.shuffle(group_indices)

                TD_selected_units.iloc[group_start:group_end, col] = (
                    TD_selected_units.iloc[group_indices, col].to_numpy()
                )
                PC_selected_units.iloc[group_start:group_end, col] = (
                    PC_selected_units.iloc[group_indices, col].to_numpy()
                )

        # adding label to dataset
        TD_selected_units['label'] = 0
        PC_selected_units['label'] = 0
        rep = 8

        for i in range(num_stimuli):
            start_index = i * rep
            end_index = start_index + rep - 1
            TD_selected_units.loc[start_index:end_index, 'label'] = i
            PC_selected_units.loc[start_index:end_index, 'label'] = i

        random_TD_selected_units = TD_selected_units.copy()
        random_PC_selected_units = PC_selected_units.copy()

        random_TD_selected_units['label'] = np.random.permutation(TD_selected_units['label'].values)
        random_PC_selected_units['label'] = np.random.permutation(PC_selected_units['label'].values)

        # cross validation
        set_index = {0, 1, 2, 3}
        map_list = [None] * (max(set_index) * 2 + 2)

        for i in set_index:

            train_index = set_index - {i}
            test_index = {i}

            for idx in test_index:
                map_list[idx * 2] = False
                map_list[idx * 2 + 1] = False

            for idx in train_index:
                map_list[idx * 2] = True
                map_list[idx * 2 + 1] = True

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

           
            # Standardize the data TD_TD
            scaler_TD = StandardScaler().fit(X_train_TD)
            X_train_TD_scaled = scaler_TD.transform(X_train_TD)
            X_test_TD_scaled = scaler_TD.transform(X_test_TD)
           
            results_TD_TD[i] = train_model_svc(
                X_train_TD_scaled,
                X_test_TD_scaled,
                y_train_TD,
                y_test_TD
            )
          

            # Standardize the data PC_PC
            scaler_PC = StandardScaler().fit(X_train_PC)
            X_train_PC_scaled = scaler_PC.transform(X_train_PC)
            X_test_PC_scaled = scaler_PC.transform(X_test_PC)
           

            results_PC_PC[i] = train_model_svc(
                X_train_PC_scaled,
                X_test_PC_scaled,
                y_train_PC,
                y_test_PC
            )
           

        ####### TD_TD
        accuracy_TD_TD = []

        for j in range(len(results_TD_TD)):
            accuracy_TD_TD.append(results_TD_TD[j]['accuracy'])
        mean_accuracy_TD_TD = np.mean(accuracy_TD_TD)
        all_mean_accuracy_TD_TD.append(mean_accuracy_TD_TD)

        ####### PC_PC
        accuracy_PC_PC = []

        for j in range(len(results_PC_PC)):
            accuracy_PC_PC.append(results_PC_PC[j]['accuracy'])
        mean_accuracy_PC_PC = np.mean(accuracy_PC_PC)   
        all_mean_accuracy_PC_PC.append(mean_accuracy_PC_PC)

    pd.Series(all_mean_accuracy_TD_TD, name="value").to_csv(
        f"{data_name}_all_mean_accuracy_original_data_TD_TD.csv", index=False
    )
    
    pd.Series(all_mean_accuracy_PC_PC, name="value").to_csv(
        f"{data_name}_all_mean_accuracy_original_data_PC_PC.csv", index=False
    )


    return {
        "all_mean_accuracy_original_data_TD_TD": all_mean_accuracy_TD_TD,
        "all_mean_accuracy_original_data_PC_PC": all_mean_accuracy_PC_PC,
    }