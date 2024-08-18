''' Python script for knn sequences comparison '''

import sys, os
import utils
import numpy as np
import utils_get_embeddings
import getopt, pickle
import scipy.stats as stats
import pandas as pd
import functools as ft

model_layer_plm_repr = {'msa':12,'esm2':48,'pt':24}

# help function
def help():
    print("Incorrect or Incomplete command line arguments")
    print('python homology_analysis.py -a alignment file -m model name -c Y')
    exit()

''' jaccard similarity '''
def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
     
    return round(intersection / union,2)

''' closest Knn sequences '''
def closest_seq(lg_1d,k,s):
    lg_1d[lg_1d == 0] = 'nan'
    k_idx = np.argpartition(lg_1d,k)
    k_nearest_idx = k_idx[:k]
    k_nearest_values = lg_1d[k_nearest_idx]
    #mean_d = round(np.mean(k_nearest_values),4)

    return [s,list(k_nearest_idx)]

''' choose K nearest sequences '''
def get_knn_sequences(k,lg_np,rand_seqs):
    knn_info_lst = []
    for s in rand_seqs:
        #print(s)
        knn_info_lst.append(closest_seq(lg_np[s],k,s))

    return knn_info_lst 
    
def groups_ji(x):
    if x == 0:
        gr = 'opposite'
    elif x == 1:
        gr = 'perfect'
    elif x > 0 and x <= 0.5:
        gr = 'low'
    elif x > 0.5 and x< 1:
        gr = 'high'
    else:
        gr = 'nogroup'

    return gr

''' function for main for loop'''
def local_sim_analysis(fasta_aln_file,model_type,k):

    file_location  = os.path.dirname(fasta_aln_file) 
    base_file_name = os.path.splitext(os.path.basename(fasta_aln_file))[0]

    fasta_file_wo_gap  = f'{file_location}/{base_file_name}_nogap.aln'
    fasta_file_std_gap = f'{file_location}/{base_file_name}_stdgap.aln'

    lg_mat_file = f'{file_location}/lg_mat.npy'
    lg_mat_np = np.load(lg_mat_file)

    seq_ref_dict_file       = f'{file_location}/sequence_ref_dict.pkl'
    file_pick = open(seq_ref_dict_file, 'rb')
    seq_ref_dict = pickle.load(file_pick)
    file_pick.close()

    # get pLM representation
    layers = [model_layer_plm_repr[model_type]]
    print(f"Getting pLM representation for protein family {base_file_name} from model {utils_get_embeddings.full_model_name[model_type]}")
    PLM_dataset = utils_get_embeddings.get_plm_representation(model_type,fasta_file_wo_gap,fasta_file_std_gap,layers)
    print(f"pLM embedding created for {len(PLM_dataset[:][0])} sequences each of size {PLM_dataset[0][1].shape}")

    # get euclidean distance
    layer_embedding = PLM_dataset[:][1][:,0,:]
    plm_seq_labels  = PLM_dataset[:][0]
    plm_seq_labels_dict = {label:idx for idx,label in enumerate(plm_seq_labels)}

    plm_mat_np = utils.create_euclidean_distance_matrix(plm_seq_labels_dict,layer_embedding,seq_ref_dict)
    print(f"pLM matrix created of shape {plm_mat_np.shape}")

    # random sequence indexes
    draw_seq = lg_mat_np.shape[0]
    rand_seqs = list(np.random.choice(len(seq_ref_dict), size = draw_seq, replace=False))
    print(f"Randomly selected {len(rand_seqs)} sequences.")
    
    knn_lg  = get_knn_sequences(k,lg_mat_np,rand_seqs)
    knn_plm = get_knn_sequences(k,plm_mat_np,rand_seqs)

    knn_lg_df = pd.DataFrame(knn_lg,columns=['ref_seq_idx','lg_knn_seq'])
    knn_plm_df = pd.DataFrame(knn_plm,columns=['ref_seq_idx','plm_knn_seq'])
    
    dfs = [knn_lg_df,knn_plm_df]
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='ref_seq_idx'), dfs)

    # JS
    df_final['plm_js'] = df_final.apply(lambda x: \
                                        jaccard_similarity(x.lg_knn_seq,x.plm_knn_seq),axis=1) 
    
    df_final['plm_js_group'] = df_final['plm_js'].apply(lambda x: groups_ji(x))
    df_final = df_final.groupby(['plm_js_group'])['plm_js'].count().reset_index()
    df_final['percentage'] = round((df_final['plm_js'] / knn_lg_df.shape[0]) * 100,2)
    return df_final

if __name__ == "__main__":

    argv = sys.argv[1:]
    opts, _ = getopt.getopt(argv, "a:m:k:")

    if len(opts) < 2:
        help()
    
    for opt, arg in opts:
        if opt in ['-a']:
            fasta_aln_file = arg
        if opt in ['-m']:
            model_type = arg # esm2, pt, msa
        if opt in ['-k']:
            k = int(arg)
    
    df_final = local_sim_analysis(fasta_aln_file,model_type,k)
    print(f"Local homolog similarity analysis complete")
    print(df_final.head())