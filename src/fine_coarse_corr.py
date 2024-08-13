''' Python script for correlation analysis for different grain of evolution time scale'''

import sys, os
import utils
import numpy as np
import utils_get_embeddings
import getopt, pickle
import scipy.stats as stats

model_layer_plm_repr = {'msa':12,'esm2':48,'pt':24}

# help function
def help():
    print("Incorrect or Incomplete command line arguments")
    print('python homology_analysis.py -a alignment file -m model name -c Y')
    exit()
    
def get_sparse_idx(a): # collect sparse sequences wrt to each sequence
    even_space_indx = list(np.linspace(0,len(a) - 2, num=10, dtype=int))
    a_sort_idx = np.argsort(a)
    index_collect = a_sort_idx[even_space_indx]
    return index_collect

def get_closest_idx(a): # collect sparse sequences wrt to each sequence
    m = np.argsort(a)[0:10]
    return m

def sparse_corr_calc(lg_a,plm_b,type_sparse_corr):
    if type_sparse_corr == 'sparse':
        lg_a[lg_a == 0] = 'nan'
        mask_array = np.apply_along_axis(get_sparse_idx,-1,lg_a)
    elif type_sparse_corr == 'near':
        lg_a[lg_a == 0] = 'nan'
        mask_array = np.apply_along_axis(get_closest_idx,-1,lg_a)

    sparse_lg_a  = np.take_along_axis(lg_a,  mask_array, axis=1)
    sparse_plm_b = np.take_along_axis(plm_b, mask_array, axis=1)

    sr_list = []
    for i in range(0,sparse_lg_a.shape[0]):
        sr = stats.pearsonr(sparse_lg_a[i],sparse_plm_b[i])[0]
        sr_list.append(sr)

    return round(np.average(sr_list),2)

def avg_corr_calc(lg_a,plm_b):
    sr_list = []
    for i in range(0,lg_a.shape[0]):
        sr = stats.pearsonr(lg_a[i],plm_b[i])[0]
        sr_list.append(sr)
    return round(np.average(sr_list),2)

def evol_scale_correlation_analysis(fasta_aln_file,model_type,colattn):

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
    if colattn == 'Y' and model_type == 'MSA': # for column attention
        print(f"Getting column attention (layer 1 head 5) for protein family {base_file_name} from model {utils_get_embeddings.full_model_name[model_type]}")
        col_attn_np = utils_get_embeddings.get_col_attn(fasta_file_std_gap,seq_ref_dict)
        col_attn_np = col_attn_np[0,4,:,:] # layer 1 head 5
        print(f"Column attention matrix created of shape {col_attn_np.shape}")
        lg_corr_avg    = avg_corr_calc(lg_mat_np,col_attn_np)
        lg_corr_sparse = sparse_corr_calc(lg_mat_np,col_attn_np,'sparse')  
        lg_corr_near   = sparse_corr_calc(lg_mat_np,col_attn_np,'near')
    
    else:
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

        lg_corr_avg    = avg_corr_calc(lg_mat_np,plm_mat_np)
        lg_corr_sparse = sparse_corr_calc(lg_mat_np,plm_mat_np,'sparse')  
        lg_corr_near   = sparse_corr_calc(lg_mat_np,plm_mat_np,'near')

    return lg_corr_avg,lg_corr_sparse,lg_corr_near

if __name__ == "__main__":

    argv = sys.argv[1:]
    opts, _ = getopt.getopt(argv, "a:m:c:")

    if len(opts) < 2:
        help()
    
    for opt, arg in opts:
        if opt in ['-a']:
            fasta_aln_file = arg
        if opt in ['-m']:
            model_type = arg # esm2, pt, msa
        if opt in ['-c']:
            colattn = arg # Y / N (only applicable for msa)
    
    lg_corr_avg,lg_corr_sparse,lg_corr_near = evol_scale_correlation_analysis(fasta_aln_file,model_type,colattn)
    print(f"Evolutionary scale analysis: Fine - {lg_corr_near}, Coarse - {lg_corr_sparse} against Average - {lg_corr_avg}")