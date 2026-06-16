#null disstribution _ multi processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import random
from collections import defaultdict
from scipy.io import loadmat
from multiprocessing import Pool
import pandas as pd
import logging
import os



def train_model_svc(X_train, X_test, y_train, y_test):
    base_svm = svm.SVC(kernel='linear', C=1)
    ovr_model = OneVsRestClassifier(base_svm)

    ovr_model.fit(X_train, y_train)
    y_pred = ovr_model.predict(X_test)

    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred)
    return results




def run_for_one_bin(dataset_id , data_control , data_main ):
    logging.basicConfig(
        filename="output/Cuba_ASB_log.txt",
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        filemode="a",
        force=True,
    )
    logging.info(f"Started dataset {dataset_id}")
    #print(f"Started dataset {dataset_id}")
    
    #original categorization
    data_name = "Cuba_ASB"
    data_control = data_control         
    data_main = data_main 

    all_500_mean_accuracy_control_random = []
    all_500_std_accuracy_control_random = []
        
    all_500_mean_accuracy_main_random = []
    all_500_std_accuracy_main_random = []
    
    for st in range(500):
        print("--------------------stage:-------------------------", st)
    
        all_mean_accuracy_control_random = []
        all_std_accuracy_control_random = []
        
        all_mean_accuracy_main_random = []
        all_std_accuracy_main_random = []
        
        all_selected_units = []
        
        num_stimuli = 40
        
        data_control['label'] = 0
        data_main['label'] = 0
        rep = 9
        #Climbing
        data_control.loc[0:44, 'label'] = 1
        data_main.loc[0:44, 'label'] = 1
        #Cornered
        data_control.loc[45:89, 'label'] = 1
        data_main.loc[45:89, 'label'] = 1
        #Eating
        data_control.loc[90:134, 'label'] = 2
        data_main.loc[90:134, 'label'] = 2
        #Jumping
        data_control.loc[135:179, 'label'] = 1
        data_main.loc[135:179, 'label'] = 1
        #Laying
        data_control.loc[180:224, 'label'] = 1
        data_main.loc[180:224, 'label'] = 1
        #Sitting / Sanding / Walking
        data_control.loc[225:359, 'label'] = 2
        data_main.loc[225:359, 'label'] = 2

        ################################################## Randomness #######################################

        group_size = 9
        n_groups   = 40
        
        # CONTROL
        #rand_control_cat1 = control_selected_units_cat1.copy()
        #rand_control_cat2 = control_selected_units_cat2.copy()
        random_control_two_cats = data_control.copy()

        labels_control = [1]*20 + [2]*20
        random.shuffle(labels_control)
        
        full_labels_control = np.repeat(labels_control, group_size)  # length 270
        random_control_two_cats['label'] = full_labels_control

         # MAIN
        #rand_main_cat1 = main_selected_units_cat1.copy()
        #rand_main_cat2 = main_selected_units_cat2.copy() 
        random_main_two_cats = data_main.copy()

        labels_main = [1]*20 + [2]*20
        random.shuffle(labels_main)
        #print(labels_main)
        
        full_labels_main = np.repeat(labels_main, group_size)
        random_main_two_cats['label'] = full_labels_main

        #print(random_main_two_cats)
        
       

        #sampling
        for sample in range(100):
            results_control_control = {}
            results_control_random = {}
            results_main_main = {}
            results_main_random = {}
        
            results_control_control_pairs = {}
            results_main_main_pairs = {}
            
            
            print("iteration: " ,sample, "for bin", dataset_id , "for stage:" , st)

            label_col = "label"  # or whatever your label column is called
            feature_cols = [c for c in data_control.columns if c != label_col]
            selected_features = random.sample(feature_cols, 50)
            selected_columns = selected_features + [label_col]   # label at the end
                       
            #selected_columns = random.sample(data_control.columns.tolist(), 50)
            #selected_columns = random.choices(data_control.columns.tolist(), k=50)
            control_selected_units = data_control.loc[:, selected_columns].copy()
            main_selected_units    = data_main.loc[:, selected_columns].copy()
            random_control_selected_units =  random_control_two_cats.loc[:, selected_columns].copy() 
            random_main_selected_units =  random_main_two_cats.loc[:, selected_columns].copy() 

                
        
            #shuffeling
            num_rows , num_columns = control_selected_units.shape 
            group_size = 9
            
            for col in range(num_columns):  
                for group_start in range(0, num_rows, group_size):  
                    group_end = min(group_start + group_size, num_rows) 
                    group_indices = np.arange(group_start, group_end)  
                    np.random.shuffle(group_indices)
                    control_selected_units.iloc[group_start:group_end, col] = control_selected_units.iloc[group_indices, col].to_numpy()
                    main_selected_units.iloc[group_start:group_end, col] = main_selected_units.iloc[group_indices, col].to_numpy()
                
                    #print(main_selected_units)
            
            #control_selected_units.to_csv("control_selected_units.csv", index = False)
           
            
            #seprate categories data into two groups for cross validation
            control_selected_units_cat1 = control_selected_units[control_selected_units['label'] == 1]
            control_selected_units_cat2 = control_selected_units[control_selected_units['label'] == 2]
        
            main_selected_units_cat1 = main_selected_units[main_selected_units['label'] == 1]
            main_selected_units_cat2 = main_selected_units[main_selected_units['label'] == 2]
        
             ################################################## Randomness #######################################

           
            random_control_cat1 = random_control_selected_units[random_control_selected_units['label'] == 1]
            random_control_cat2 = random_control_selected_units[random_control_selected_units['label'] == 2]
           
            random_main_cat1 = random_main_selected_units[random_main_selected_units['label'] == 1]
            random_main_cat2 = random_main_selected_units[random_main_selected_units['label'] == 2]

            #############################################################################################
            
            #cross validation   
            set_index = [0, 1, 2, 3]     # list, not set
            cat_num = 8
            view = 5
            rep = 9
            fold=0
            map_list_1 = [None] * (max(set_index) * (rep * view) + (rep * view))  # 4 * 45 = 180
            map_list_2 = [None] * (max(set_index) * (rep * view) + (rep * view))  # 4 * 45 = 180
            
            shuffled_set_index_1 = np.random.permutation(set_index).tolist()  # e.g., [1, 2, 0, 3]
            shuffled_set_index_2 = np.random.permutation(set_index).tolist()  # e.g., [1, 2, 0, 3]
            
            #print("sh1",shuffled_set_index_1)
            #print("sh2",shuffled_set_index_2)
            
            for i, j in zip(shuffled_set_index_1, shuffled_set_index_2):
                
                #print("fold:", fold)
                #print("pair:", i, j)
                
                train_index_1 = [x for x in set_index if x != i]  # list version of set difference
                test_index_1  = [i]
                
                train_index_2 = [x for x in set_index if x != j]  # list version of set difference
                test_index_2  = [j]
            
                for idx in test_index_1:
                    map_list_1[idx*(rep * view) : (idx+1)*(rep * view)] = [False] * rep * view
                for idx in test_index_2:
                    map_list_2[idx*(rep * view) : (idx+1)*(rep * view)] = [False] * rep * view
            
                for idx in train_index_1:
                    map_list_1[idx*(rep * view) : (idx+1)*(rep * view)] = [True] * rep * view
                for idx in train_index_2:
                    map_list_2[idx*(rep * view) : (idx+1)*(rep * view)] = [True] * rep * view
                
                #print("map_list1", len(map_list_1) , map_list_1)
                train_set_control_cat1 = control_selected_units_cat1[(map_list_1)]
                test_set_control_cat1 = control_selected_units_cat1[np.logical_not(map_list_1)]
                
                train_set_main_cat1 = main_selected_units_cat1[(map_list_1)]
                test_set_main_cat1 = main_selected_units_cat1[np.logical_not(map_list_1)]

                #random set
                train_set_random_control_cat1 = random_control_cat1[(map_list_1)]            
                test_set_random_control_cat1 = random_control_cat1[np.logical_not(map_list_1)]

                train_set_random_main_cat1 = random_main_cat1[(map_list_1)]            
                test_set_random_main_cat1 = random_main_cat1[np.logical_not(map_list_1)]
            
           
                #print("map_list2", len(map_list_2) , map_list_2)
                train_set_control_cat2 = control_selected_units_cat2[(map_list_2)]
                test_set_control_cat2 = control_selected_units_cat2[np.logical_not(map_list_2)] 
                
                train_set_main_cat2 = main_selected_units_cat2[(map_list_2)]
                test_set_main_cat2 = main_selected_units_cat2[np.logical_not(map_list_2)]

                #random set
                train_set_random_control_cat2 = random_control_cat2[(map_list_2)]            
                test_set_random_control_cat2 = random_control_cat2[np.logical_not(map_list_2)]

                train_set_random_main_cat2 = random_main_cat2[(map_list_2)]            
                test_set_random_main_cat2 = random_main_cat2[np.logical_not(map_list_2)]
            
        
        
                train_set_control = pd.concat([train_set_control_cat1 , train_set_control_cat2], axis = 0)
                #train_set_control.to_csv(f"train_set_control_{fold}.csv", index=False)
                #train_set_control = train_set_control.drop(columns =['label_stimuli'])
               
                test_set_control = pd.concat([test_set_control_cat1 , test_set_control_cat2], axis = 0)    
                #test_set_control.to_csv(f"test_set_control_{fold}.csv", index=False)
               # test_set_control = test_set_control.drop(columns =['label_stimuli'])
        
                
                train_set_main = pd.concat([train_set_main_cat1 , train_set_main_cat2], axis = 0)    
                test_set_main = pd.concat([test_set_main_cat1 , test_set_main_cat2], axis = 0)
               # train_set_main = train_set_main.drop(columns = ['label_stimuli'])
                #test_set_main = test_set_main.drop(columns = ['label_stimuli'])

                ############################################## Randomness #####################################
                ###random set
                train_set_random_control = pd.concat([train_set_random_control_cat1 , train_set_random_control_cat2], axis = 0)
                #train_set_random_control = train_set_random_control.drop(columns =['label_stimuli'])
                test_set_random_control = pd.concat([test_set_random_control_cat1 , test_set_random_control_cat2], axis = 0)
                #test_set_random_control = test_set_random_control.drop(columns =['label_stimuli'])

                train_set_random_main = pd.concat([train_set_random_main_cat1 , train_set_random_main_cat2], axis = 0)
                #train_set_random_main = train_set_random_main.drop(columns =['label_stimuli'])
                test_set_random_main = pd.concat([test_set_random_main_cat1 , test_set_random_main_cat2], axis = 0)
               # test_set_random_main = test_set_random_main.drop(columns =['label_stimuli'])

                
                #train_set_random_control['label'] = np.random.permutation(train_set_control['label'].values)
                #train_set_random_main['label'] = np.random.permutation(train_set_random_main['label'].values)
               
        
                X_train_control = train_set_control.iloc[:, :-1] 
                y_train_control = train_set_control['label']
        
                X_test_control = test_set_control.iloc[:, :-1]
                y_test_control = test_set_control['label']
        
        
                X_train_main = train_set_main.iloc[:, :-1] 
                y_train_main = train_set_main['label']
        
                X_test_main = test_set_main.iloc[:, :-1]
                y_test_main = test_set_main['label']
                
        
                X_train_control_random = train_set_random_control.iloc[:, :-1] 
                y_train_control_random = train_set_random_control['label']
        
                #X_test_control_random = test_set_random_control.iloc[:, :-1]
                #y_test_control_random = test_set_random_control['label']
        
                X_train_main_random = train_set_random_main.iloc[:, :-1] 
                y_train_main_random = train_set_random_main['label']
        
               # X_test_main_random = test_set_random_main.iloc[:, :-1]
               # y_test_main_random = test_set_random_main['label']
                
        
                
                scaler_control = StandardScaler().fit(X_train_control)
                X_train_control_scaled = scaler_control.transform(X_train_control)
                X_test_control_scaled  = scaler_control.transform(X_test_control)
                X_train_control_random_scaled = scaler_control.transform(X_train_control_random)
                #X_test_control_random_scaled     = scaler_control.transform(X_test_control_random)
        
              #  print("---------Accuracy Control_Control-----------")        
                #result_control = train_model_svc(X_train_control_scaled, X_test_control_scaled, y_train_control, y_test_control )
                #results_control_control[fold] = result_control
                #results_control_control_pairs[(i, j)] = result_control
                #print("---------Accuracy Control_Random-----------")
                results_control_random[fold] = train_model_svc(X_train_control_random_scaled, X_test_control_scaled, y_train_control_random, y_test_control )
               # print(results_control_random)
            
                ####### Main_control
        
                scaler_main = StandardScaler().fit(X_train_main)
                X_train_main_scaled = scaler_main.transform(X_train_main)
                X_test_main_scaled  = scaler_main.transform(X_test_main)
                X_train_main_random_scaled = scaler_main.transform(X_train_main_random)
               # X_test_main_random_scaled  = scaler_main.transform(X_test_main_random)
        
               # print("---------Accuracy Main_Main-----------")            
                #result_main = train_model_svc(X_train_main_scaled, X_test_main_scaled, y_train_main, y_test_main )
                #results_main_main[fold] = result_main
                #results_main_main_pairs[(i, j)] = result_main
                
                #print("---------Accuracy Main_Random-----------")
                results_main_random[fold] = train_model_svc(X_train_main_random_scaled, X_test_main_scaled, y_train_main_random, y_test_main )
                #print(results_main_random)
                fold+=1
        
                
        
            ####### control_random
           
            accuracy_control_random=[]
        
            for fold_acc in results_control_random:
                accuracy_control_random.append(results_control_random[fold_acc]['accuracy'])
          
            mean_accuracy_control_random = np.mean(accuracy_control_random)           
            all_mean_accuracy_control_random.append(mean_accuracy_control_random)
        
        
            ####### Main_random
        
            accuracy_main_random=[] 
            
            for fold_cc_main in range(len(results_main_random)):
                accuracy_main_random.append(results_main_random[fold_cc_main]['accuracy'])
         
            mean_accuracy_main_random = np.mean(accuracy_main_random)        
            all_mean_accuracy_main_random.append(mean_accuracy_main_random)
        
            all_selected_units.append(selected_columns)
            
    
        all_500_mean_accuracy_control_random.append(np.mean(all_mean_accuracy_control_random) )
        all_500_std_accuracy_control_random.append(np.std(all_mean_accuracy_control_random)) 

        all_500_mean_accuracy_main_random.append(np.mean(all_mean_accuracy_main_random)) 
        all_500_std_accuracy_main_random.append(np.std(all_mean_accuracy_main_random) )

    prefix = f"Cuba_ASB_dataset_{dataset_id}"
    save_results(
        prefix,
        all_500_mean_accuracy_control_random,
        all_500_mean_accuracy_main_random
    )
    # optionally return summary
    logging.info(f"Finished dataset {dataset_id}")
    return dataset_id 

    

def save_results(prefix,
                 all_500_mean_accuracy_control_random,
                 all_500_mean_accuracy_main_random):

    df_mean_ctrl = pd.DataFrame(all_500_mean_accuracy_control_random)
    df_mean_main = pd.DataFrame(all_500_mean_accuracy_main_random)

    df_mean_ctrl.to_csv(f"output/{prefix}_all_500_mean_accuracy_control_random.csv",
                        index=False)
    df_mean_main.to_csv(f"output/{prefix}_all_500_mean_accuracy_main_random.csv",
                        index=False)
    
    




if __name__ == "__main__":

    
    data_control_medium_bins = loadmat("G:/PHD-proj/WP2/Matlab/data_for_analysis - occluded/Cuba_all_data_res_window_bins_svm2_control_ASB.mat")
    data_main_medium_bins = loadmat("G:/PHD-proj/WP2/Matlab/data_for_analysis - occluded/Cuba_all_data_res_window_bins_svm2_main_ASB.mat")


    data_control_bins = data_control_medium_bins ['all_data_res_window_bins_svm2_control'] 
    data_main_bins = data_main_medium_bins['all_data_res_window_bins_svm2_main']  
    # Example: list of arguments for each dataset
    datasets_args = [
        (0, pd.DataFrame(data_control_bins[:,0,:]), pd.DataFrame(data_main_bins[:,0,:])),
        (1, pd.DataFrame(data_control_bins[:,1,:]), pd.DataFrame(data_main_bins[:,1,:])),
        (2, pd.DataFrame(data_control_bins[:,2,:]), pd.DataFrame(data_main_bins[:,2,:])),
        (3, pd.DataFrame(data_control_bins[:,3,:]), pd.DataFrame(data_main_bins[:,3,:])),
        (4, pd.DataFrame(data_control_bins[:,4,:]), pd.DataFrame(data_main_bins[:,4,:])),
        (5, pd.DataFrame(data_control_bins[:,5,:]), pd.DataFrame(data_main_bins[:,5,:])),
        (6, pd.DataFrame(data_control_bins[:,6,:]), pd.DataFrame(data_main_bins[:,6,:])),
        (7, pd.DataFrame(data_control_bins[:,7,:]), pd.DataFrame(data_main_bins[:,7,:])),
        (8, pd.DataFrame(data_control_bins[:,8,:]), pd.DataFrame(data_main_bins[:,8,:])),
        (9, pd.DataFrame(data_control_bins[:,9,:]), pd.DataFrame(data_main_bins[:,9,:])),
        (10, pd.DataFrame(data_control_bins[:,10,:]), pd.DataFrame(data_main_bins[:,10,:])),
        (11, pd.DataFrame(data_control_bins[:,11,:]), pd.DataFrame(data_main_bins[:,11,:])),
        (12, pd.DataFrame(data_control_bins[:,12,:]), pd.DataFrame(data_main_bins[:,12,:])),
        (13, pd.DataFrame(data_control_bins[:,13,:]), pd.DataFrame(data_main_bins[:,13,:])),
        (14, pd.DataFrame(data_control_bins[:,14,:]), pd.DataFrame(data_main_bins[:,14,:])),
        (15, pd.DataFrame(data_control_bins[:,15,:]), pd.DataFrame(data_main_bins[:,15,:])),
        (16, pd.DataFrame(data_control_bins[:,16,:]), pd.DataFrame(data_main_bins[:,16,:])),
        (17, pd.DataFrame(data_control_bins[:,17,:]), pd.DataFrame(data_main_bins[:,17,:])),
        (18, pd.DataFrame(data_control_bins[:,18,:]), pd.DataFrame(data_main_bins[:,18,:])),
        (19, pd.DataFrame(data_control_bins[:,19,:]), pd.DataFrame(data_main_bins[:,19,:])),
        (20, pd.DataFrame(data_control_bins[:,20,:]), pd.DataFrame(data_main_bins[:,20,:])),
        (21, pd.DataFrame(data_control_bins[:,21,:]), pd.DataFrame(data_main_bins[:,21,:])),
        (22, pd.DataFrame(data_control_bins[:,22,:]), pd.DataFrame(data_main_bins[:,22,:])),
        (23, pd.DataFrame(data_control_bins[:,23,:]), pd.DataFrame(data_main_bins[:,23,:])),
        (24, pd.DataFrame(data_control_bins[:,24,:]), pd.DataFrame(data_main_bins[:,24,:])),
       
    ]

    #for args in datasets_args[:2]:   # try first 2 bins
    #    run_for_one_bin(*args)
    n_workers = min(13, os.cpu_count() - 1)
    
    with Pool(processes=n_workers) as pool:  # run 5 datasets in parallel
        results = pool.starmap(run_for_one_bin, datasets_args)
        
    print(f"Completed datasets: {results}", flush=True)