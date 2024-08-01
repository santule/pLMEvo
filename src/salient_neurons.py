''' Python script for correlation analysis for different grain of evolution time scale'''

import sys, os
import utils
import numpy as np
import utils_get_embeddings
import getopt, pickle
import scipy.stats as stats
import pandas as pd
import elastic_net_utils

model_layer_plm_repr = {'msa':12,'esm2':48,'pt':24}

# help function
def help():
    print("Incorrect or Incomplete command line arguments")
    print('python homology_analysis.py -a alignment file -m model name')
    exit()


def salient_neurons(fasta_aln_file,model_type,n_samples_per_class = 3000):

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

    # prepare the data for the probe
    train_data_df = elastic_net_utils.prep_training_data(PLM_dataset,lg_mat_np,seq_ref_dict)
    print(f"Pre-assembled training data of shape - {train_data_df.shape}")

    # remove low varaince features
    train_nueron_df = elastic_net_utils.remove_low_variance_features(train_data_df)
    print(f"Assembled training data after filtering low variance neurons - {train_nueron_df.shape}")
    
    #find the hyperparameters for logistic regression
    print(f"Finding best hyperparameters for elastic net")
    best_alpha,best_l1_ratio = elastic_net_utils.elastic_net_hyper(train_nueron_df,n_samples_per_class)
    print(f"Best hyperparameters alpha - {best_alpha} and l1_ratio - {best_l1_ratio}")

    # run on all data to find coefficients of regression
    print(f"Running elastic net regression to find salient neurons")
    score_lst,coeff_wt_lst = elastic_net_utils.find_coeff(train_nueron_df,best_alpha,best_l1_ratio,n_samples_per_class)
        
    # get top and bottom weighted features
    top_neuron_nos_10,\
        top_neuron_nos_25,\
            top_neuron_nos_50,\
                top_neuron_nos_75,\
                    bottom_neuron_nos, zero_neuron_no = elastic_net_utils.wt_features(coeff_wt_lst,train_nueron_df.columns)
        
    # RSS using the top and bottom neurons
    print(f"RSS analysis using top 10% of ranked nuerons ")
    elastic_net_utils.get_corr_neurons('top_10',\
                                    top_neuron_nos_10,seq_ref_dict,lg_mat_np,PLM_dataset)
    print(f"RSS analysis using top 25% of ranked nuerons ")
    elastic_net_utils.get_corr_neurons('top_25',\
                                    top_neuron_nos_25,seq_ref_dict,lg_mat_np,PLM_dataset)
    print(f"RSS analysis using top 50% of ranked nuerons ")
    elastic_net_utils.get_corr_neurons('top_50',\
                                    top_neuron_nos_50,seq_ref_dict,lg_mat_np,PLM_dataset)
    print(f"RSS analysis using top 75% of ranked nuerons ")
    elastic_net_utils.get_corr_neurons('top_75',\
                                    top_neuron_nos_75,seq_ref_dict,lg_mat_np,PLM_dataset)
    print(f"RSS analysis using bottom 25% of ranked nuerons ")
    elastic_net_utils.get_corr_neurons('bottom',\
                                    bottom_neuron_nos,seq_ref_dict,lg_mat_np,PLM_dataset)


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
    
    salient_neurons(fasta_aln_file,model_type)