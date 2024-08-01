''' functions for elastic net regression to find salient neurons '''
import torch
import pandas as pd
import numpy as np
import utils
from sklearn.feature_selection import VarianceThreshold
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def draw_sample(train_data_df,class_sample,r=10):
    # returns numpy array for X and list for y
    
    # equal samples for each class
    X_train = []
    y_train = []
    evol_dist_values = ['1low', '2medium', '3high']
    
    for c in evol_dist_values:
        train_sample = train_data_df[train_data_df['y_class'] == c].sample(n=class_sample, random_state=r)
        y_train.append(list(train_sample['evol_dist']))
        X_train.append(np.array(train_sample.loc[:, ~train_sample.columns.isin(['evol_dist','y_class'])]))
            
    y_train = list(chain.from_iterable(y_train))
    X_train = np.vstack(X_train)
    
    return X_train,y_train

def elastic_net_hyper(train_data_df,n_samples):
    
    X_train,y_train = draw_sample(train_data_df,n_samples) # samples of each class
    print(f"X_train sample size - {X_train.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # elasticnetcv
    ratios = [0.5,0.9,1]
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    elastic_net_cv = ElasticNetCV(cv=10, random_state=1,max_iter=1000,l1_ratio=ratios,alphas=alphas)
    
    # Train the model
    elastic_net_cv.fit(X_train_scaled, y_train)
    
    return elastic_net_cv.alpha_,elastic_net_cv.l1_ratio_


def remove_low_variance_features(train_data_df,thres_precentile=20):
    
    neuron_np = torch.vstack(list(train_data_df['abs_distance'])).numpy()
    print(f"Assembled training data - {neuron_np.shape}")
    
    thres = np.percentile(np.var(neuron_np,axis=0), thres_precentile)
    print(f"Threshold selected for cutoff {thres}") 
    
    var_thr = VarianceThreshold(threshold=thres)
    var_thr.fit(neuron_np)
    
    # filter the neurons based on variance threshold
    neuron_exp = np.arange(0,neuron_np.shape[1])
    neuron_nu  = neuron_exp[var_thr.get_support()]
    neuron_np  = neuron_np[:,var_thr.get_support()]
    
    train_nueron_df = pd.DataFrame(neuron_np,columns = neuron_nu)
    
    # put the other columns back
    train_nueron_df['evol_dist'] = train_data_df['evol_dist']
    train_nueron_df['y_class']   = train_data_df['y_class']
    
    return train_nueron_df

def salient_features_by_threshold(ordered_index_nonzero,norm_coeff,neuron_numbers_ref,top_threshold):
    
    # top wt threshold
    wt_top_features = []
    total_wt,temp_wt = 0, 0
    for f in ordered_index_nonzero:
        temp_wt = total_wt + norm_coeff[f]
        if temp_wt > top_threshold:
            break
        else:
            total_wt = temp_wt
            wt_top_features.append(f)
    top_neuron_nos_th = neuron_numbers_ref[wt_top_features]
    return top_neuron_nos_th


def wt_features(coeff_wt_lst,neuron_numbers_ref):
    
    # norm features
    sum_coeff = np.sum(coeff_wt_lst,axis=0)
    norm_coeff = sum_coeff / sum(sum_coeff)

    # order in descending order
    ordered_index = np.argsort(-norm_coeff) 

    # remove coefficients with 0s
    zero_features         = [f for f in ordered_index if norm_coeff[f] == 0]        
    ordered_index_nonzero = [f for f in ordered_index if norm_coeff[f] != 0]
    
    top_threshold = 0.10
    top_neuron_nos_10 = salient_features_by_threshold(ordered_index_nonzero,\
                                        norm_coeff,neuron_numbers_ref,top_threshold)

    top_threshold = 0.25
    top_neuron_nos_25 = salient_features_by_threshold(ordered_index_nonzero,\
                                        norm_coeff,neuron_numbers_ref,top_threshold)

    top_threshold = 0.50
    top_neuron_nos_50 = salient_features_by_threshold(ordered_index_nonzero,\
                                        norm_coeff,neuron_numbers_ref,top_threshold)

    top_threshold = 0.75
    top_neuron_nos_75 = salient_features_by_threshold(ordered_index_nonzero,\
                                        norm_coeff,neuron_numbers_ref,top_threshold)

    # bottom wt threshold
    bottom_threshold = 0.25
    wt_bottom_features = []
    total_wt,temp_wt = 0, 0
    for f in ordered_index_nonzero[::-1]:
        temp_wt = total_wt + norm_coeff[f]
        if temp_wt > bottom_threshold:
            break
        else:
            total_wt = temp_wt
            wt_bottom_features.append(f)
    
    bottom_neuron_nos = neuron_numbers_ref[wt_bottom_features]
    zero_neuron_no    = neuron_numbers_ref[zero_features]

    
    return top_neuron_nos_10,top_neuron_nos_25,top_neuron_nos_50,top_neuron_nos_75,bottom_neuron_nos, zero_neuron_no


def find_coeff(train_data_df,best_alpha,best_l1_ratio,n_samples):
    
    # Define the model pipeline
    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,max_iter=1000,selection = 'random')
    model_pipeline = Pipeline([ ('scale', StandardScaler()),
                                ('en', model)])
    
    # fit the model on 10 random samples of data in the paper, here we run for 2.
    score_lst,features_coeff_lst = [], []

    for r in [0,45]:
        print(f"Running classifier with random seed {r}")
        X_train,y_train = draw_sample(train_data_df,n_samples,r)
        print(f"Assembled training data - X_train- {X_train.shape} and y_train- {len(y_train)}")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.30,random_state=42)

        # Train the model and evaluate
        r2_score = model_pipeline.fit(X_train,y_train).score(X_test,y_test)
        score_lst.append(np.round(r2_score,2))
        features_coeff_lst.append(np.abs(model.coef_))

    print(f"Completed regression for 2 random seeds")
    return score_lst,features_coeff_lst


def calc_abs_distance(a,b): # a - (1,d) vector
  return np.abs(a-b).reshape(-1)

def prep_training_data(PLM_dataset,lg_matrix,seq_ref_dict):

    train_data = []
    # get euclidean distance
    plm_matrix = PLM_dataset[:][1][:,0,:]
    plm_seq_labels  = PLM_dataset[:][0]
    plm_seq_labels_dict = {label:idx for idx,label in enumerate(plm_seq_labels)}
    plm_seq_name_list = list(plm_seq_labels_dict.keys())

    ######## PAIRWISE DISTANCE #####
    for ref_num, ref_extant_sequence in enumerate(plm_seq_name_list):
        for other_extant_sequence in plm_seq_name_list[ref_num + 1:]:

            # universal ids in the universal list
            uni_ref_idx   = seq_ref_dict[ref_extant_sequence]
            uni_other_idx = seq_ref_dict[other_extant_sequence]

            # sequence embedding
            ref_seq_embedding   = plm_matrix[plm_seq_labels_dict[ref_extant_sequence]]
            other_seq_embedding = plm_matrix[plm_seq_labels_dict[other_extant_sequence]]
            dim_dist  = calc_abs_distance(ref_seq_embedding,other_seq_embedding)

            # lg distance
            pred_dist = lg_matrix[uni_ref_idx][uni_other_idx]
            train_data.append([ref_extant_sequence + '::::' + other_extant_sequence,dim_dist,pred_dist])

    train_data_df = pd.DataFrame(train_data,columns=['seq_pair','abs_distance','evol_dist'])
    
    # categorise the evol distance into classes
    y_train = np.array(train_data_df['evol_dist'])    
    small,  high = np.percentile(y_train, 25) ,np.percentile(y_train, 75) 
    
    dist_conditions = [
    (train_data_df['evol_dist'] <= small),
    (train_data_df['evol_dist'] > small) & (train_data_df['evol_dist'] < high) ,
    (train_data_df['evol_dist'] >= high)
    ]
    
    # create a list of the values we want to assign for each condition
    evol_dist_values = ['1low', '2medium', '3high']
    
    # create a new column and use np.select to assign values to it using our lists as arguments
    train_data_df['y_class'] = np.select(dist_conditions, evol_dist_values)
    train_data_df = train_data_df.loc[train_data_df['y_class'].isin(evol_dist_values)]

    return train_data_df


def get_corr_neurons(neuron_type,filter_neuron,seq_ref_dict,lg_mat_np,PLM_dataset):

    plm_seq_labels = PLM_dataset[:][0]
    plm_matrix     = PLM_dataset[:][1].squeeze(1)
    print(f"orginial plm_matrix shape {plm_matrix.shape}")

    plm_matrix_filter = plm_matrix[:,filter_neuron].squeeze(1)
    print(f"salient neurons fitlered plm_matrix shape {plm_matrix_filter.shape}")

    plm_seq_labels_dict = {label:idx for idx,label in enumerate(plm_seq_labels)}
    plm_euc_np = utils.create_euclidean_distance_matrix(plm_seq_labels_dict,\
                        plm_matrix_filter,seq_ref_dict)

    rss_magnitude = round(utils.pearson_corr(lg_mat_np,plm_euc_np)[0],4)
    rss_order = round(utils.sperman_rank_corr(lg_mat_np,plm_euc_np)[0],4)
    print(f"RSS(order) based on filtered neurons is {rss_order}")
    print(f"RSS(magnitude) based on filtered neurons is {rss_magnitude}")
           
