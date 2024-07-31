import sys, os
import utils
import numpy as np
import utils_get_embeddings
import getopt, pickle


# help function
def help():
    print("Incorrect or Incomplete command line arguments")
    print('python one_hot_analysis.py -a alignment file -m model name')
    exit()
    

def layer_wise_one_hot_corr(plm_dataset,onehot_matrix,uni_sequence_names_dict):
    rho_layer_corr,pearson_layer_corr = [], []

    plm_seq_labels = plm_dataset[:][0]
    plm_seq_labels_dict = {label:idx for idx,label in enumerate(plm_seq_labels)}
    total_layers = plm_dataset[:][1].shape[1]

    for l in range(total_layers):
        layer_embedding = plm_dataset[:][1][:,l,:]
        layer_distance = utils.create_euclidean_distance_matrix(plm_seq_labels_dict,layer_embedding,uni_sequence_names_dict)
        c1 = round(utils.sperman_rank_corr(onehot_matrix,layer_distance)[0],4)
        c2 = round(utils.pearson_corr(onehot_matrix,layer_distance)[0],4)

        rho_layer_corr.append(c1)
        pearson_layer_corr.append(c2)

    return rho_layer_corr,pearson_layer_corr

def oh_correlation_analysis(fasta_aln_file,model_type):

    file_location  = os.path.dirname(fasta_aln_file) 
    base_file_name = os.path.splitext(os.path.basename(fasta_aln_file))[0]
    fasta_file_wo_gap  = f'{file_location}/{base_file_name}_nogap.aln'
    fasta_file_std_gap = f'{file_location}/{base_file_name}_stdgap.aln'
    seq_ref_dict_file  = f'{file_location}/sequence_ref_dict.pkl'

    file_pick = open(seq_ref_dict_file, 'rb')
    seq_ref_dict = pickle.load(file_pick)
    file_pick.close()

    # assemble one hot matrix
    oh_seq_label, oh_embedding_np = utils.get_seq_oh_embedding(fasta_file_wo_gap)
    oh_seq_labels_dict = {label:idx for idx,label in enumerate(oh_seq_label)}

    one_hot_euclidean_np = utils.create_euclidean_distance_matrix(oh_seq_labels_dict, oh_embedding_np, seq_ref_dict)
    print(f"one hot euclidean distance array created {one_hot_euclidean_np.shape}")
    np.save(os.path.join(file_location ,'onehot_mat.npy'), one_hot_euclidean_np)

    # get pLM representation
    layers = 'all'
    print(f"Getting pLM representation for protein family {base_file_name} from model {utils_get_embeddings.full_model_name[model_type]}")
    PLM_dataset = utils_get_embeddings.get_plm_representation(model_type,fasta_file_wo_gap,fasta_file_std_gap,layers)
    print(f"pLM embedding created for protein family {base_file_name} for {len(PLM_dataset[:][0])} each of shape {PLM_dataset[0][1].shape}")

    # get rss
    rho_layer_corr,pearson_layer_corr = layer_wise_one_hot_corr(PLM_dataset,one_hot_euclidean_np,seq_ref_dict)
    return  rho_layer_corr,pearson_layer_corr

if __name__ == "__main__":

    argv = sys.argv[1:]
    opts, _ = getopt.getopt(argv, "a:m:")

    if len(opts) < 2:
        help()
    
    for opt, arg in opts:
        if opt in ['-a']:
            fasta_aln_file = arg
        if opt in ['-m']:
            model_type = arg # esm2, pt, msa
    
    rho_layer_corr,pearson_layer_corr = oh_correlation_analysis(fasta_aln_file,model_type)
    print(f"RSS (order): {rho_layer_corr}")
    print(f"RSS (magnitude): {pearson_layer_corr}")