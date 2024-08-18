''' Python script for correlation analysis against lg matrix for output layer of pLM'''

import sys, os
import utils
import numpy as np
import utils_get_embeddings
import getopt, pickle

model_layer_plm_repr = {'msa':12,'esm2':48,'pt':24}

# help function
def help():
    print("Incorrect or Incomplete command line arguments")
    print('python homology_analysis.py -a alignment file -m model name -s Y -c Y')
    exit()
    

def layer_wise_lg_corr(plm_dataset,lg_mat_np,uni_sequence_names_dict):
    rho_layer_corr,pearson_layer_corr = [], []

    plm_seq_labels = plm_dataset[:][0]
    plm_seq_labels_dict = {label:idx for idx,label in enumerate(plm_seq_labels)}
    total_layers = plm_dataset[:][1].shape[1]

    for l in range(total_layers):
        layer_embedding = plm_dataset[:][1][:,l,:]
        layer_distance = utils.create_euclidean_distance_matrix(plm_seq_labels_dict,layer_embedding,uni_sequence_names_dict)
        #print(f"layer_distance {layer_distance.shape}")
    
        c1 = round(utils.sperman_rank_corr(lg_mat_np,layer_distance)[0],4)
        c2 = round(utils.pearson_corr(lg_mat_np,layer_distance)[0],4)

        rho_layer_corr.append(c1)
        pearson_layer_corr.append(c2)

    return rho_layer_corr,pearson_layer_corr

def hm_correlation_analysis(fasta_aln_file,model_type,shuffle,colattn):

    file_location  = os.path.dirname(fasta_aln_file) 
    base_file_name = os.path.splitext(os.path.basename(fasta_aln_file))[0]

    # shuffled or not    
    if shuffle == 'Y':
        print(f'Analysing shuffled fasta file')
        fasta_file_wo_gap  = f'{file_location}/{base_file_name}_nogap_0.8.aln'
        fasta_file_std_gap = f'{file_location}/{base_file_name}_stdgap_0.8.aln'
    else:
        fasta_file_wo_gap  = f'{file_location}/{base_file_name}_nogap.aln'
        fasta_file_std_gap = f'{file_location}/{base_file_name}_stdgap.aln'

    lg_mat_file = f'{file_location}/lg_mat.npy'
    lg_mat_np = np.load(lg_mat_file)

    seq_ref_dict_file       = f'{file_location}/sequence_ref_dict.pkl'
    file_pick = open(seq_ref_dict_file, 'rb')
    seq_ref_dict = pickle.load(file_pick)
    file_pick.close()

    # get pLM representation
    if colattn == 'Y' and model_type == 'msa': # for column attention
        print(f"Getting column attention (layer 1 head 5) for protein family {base_file_name} from model {utils_get_embeddings.full_model_name[model_type]}")
        col_attn_np = utils_get_embeddings.get_col_attn(fasta_file_std_gap,seq_ref_dict)
        col_attn_np = col_attn_np[0,4,:,:] # layer 1 head 5
        print(f"Column attention matrix created of shape {col_attn_np.shape}")
        rho_layer_corr = [round(utils.sperman_rank_corr(lg_mat_np,col_attn_np)[0],4)]
        pearson_layer_corr = [round(utils.pearson_corr(lg_mat_np,col_attn_np)[0],4)]

    else:
        layers = [model_layer_plm_repr[model_type]]
        print(f"Getting pLM representation for protein family {base_file_name} from model {utils_get_embeddings.full_model_name[model_type]}")
        PLM_dataset = utils_get_embeddings.get_plm_representation(model_type,fasta_file_wo_gap,fasta_file_std_gap,layers)
        print(f"pLM embedding created for {len(PLM_dataset[:][0])} sequences each of size {PLM_dataset[0][1].shape}")
        rho_layer_corr,pearson_layer_corr = layer_wise_lg_corr(PLM_dataset,lg_mat_np,seq_ref_dict)

    return  rho_layer_corr,pearson_layer_corr

if __name__ == "__main__":

    argv = sys.argv[1:]
    opts, _ = getopt.getopt(argv, "a:m:s:c:")

    if len(opts) < 4:
        help()
    
    for opt, arg in opts:
        if opt in ['-a']:
            fasta_aln_file = arg
        if opt in ['-m']:
            model_type = arg # esm2, pt, msa
        if opt in ['-s']:
            shuffle = arg
        if opt in ['-c']:
            colattn = arg # Y / N (only applicable for msa)
        
    rho_layer_corr,pearson_layer_corr = hm_correlation_analysis(fasta_aln_file,model_type,shuffle,colattn)
    print(f"RSS (order): {rho_layer_corr}")
    print(f"RSS (magnitude): {pearson_layer_corr}")