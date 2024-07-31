''' utiliy file for common functions '''

from pysam import FastaFile,FastxFile
import subprocess
import numpy as np
from ete3 import Tree
import os
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import scipy.stats as stats

allowed_aa_chars = ['.','-', 'R', 'H', 'K','D', 'E','S', 'T', 'N', 'Q','C', 'G', 'P','A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

''' function to standarise gap character '''
def std_gap_chars(extant_fasta_file,fasta_file_name_std_gap,gap_character):
    with FastxFile(extant_fasta_file, 'r') as fh, open(fasta_file_name_std_gap, 'w') as outfile:
        for entry in fh:
          outfile.write('>' + str(entry.name) + '\n')
          outfile.write(str(entry.sequence.replace(gap_character,"-"))  + '\n')
    fh.close()
    outfile.close()
    print(f"Gaps are standardised and new file {fasta_file_name_std_gap} is created")

''' function to check whether no standard amino acids present '''
def aa_check(extant_fasta_file):
  with FastxFile(extant_fasta_file, 'r') as fh:
    for entry in fh:
      for s in [*entry.sequence]:
        if s not in allowed_aa_chars:
         print(f"Found illegal aa character {s}")
         return 0
  fh.close()
  return 1

''' function to create reference sequnce dictionary '''
def create_ref_idx(fasta_file):
  seq_ref_dict = {}
  idx_no = 0
  with FastxFile(fasta_file, 'r') as fh:
    for entry in fh:
      seq_ref_dict[entry.name] = idx_no
      idx_no = idx_no + 1
  return seq_ref_dict

''' function to remove gap in the protein sequence '''
def remove_gaps(extant_fasta_file,fasta_file_name_wo_gap,gap_character):
      with FastxFile(extant_fasta_file, 'r') as fh, open(fasta_file_name_wo_gap, 'w') as outfile:
        for entry in fh:
          outfile.write('>' + str(entry.name) + '\n')
          outfile.write(str(entry.sequence.replace(gap_character,""))  + '\n')
      fh.close()
      outfile.close()
      print(f"Gaps removed and new file {fasta_file_name_wo_gap} created.")

''' function to create LG tree'''
def create_lg_tree(aln_fasta_file,nwk_file_name):
    fasttree_cmd = "FastTree -seed {rand_seed} -quiet -lg {input_file} > {output_file}".format(rand_seed=0,input_file = aln_fasta_file,output_file = nwk_file_name)
    subprocess.run(fasttree_cmd,shell=True)
    print(f"Created Phylogenetic Tree for {aln_fasta_file}.")

''' function to create sequence dictionary '''
def create_ref_idx(fasta_file):
    seq_ref_dict = {}
    idx_no = 0
    with FastxFile(fasta_file, 'r') as fh:
        for entry in fh:
            seq_ref_dict[entry.name] = idx_no
            idx_no = idx_no + 1  
    return seq_ref_dict

''' function to generate LG matrix '''
def generate_lg_matrix(seq_ref_dict,nwk_file_name):
   
    extant_sequence_names = list(seq_ref_dict.keys())
    cophenetic_dist_np    = np.zeros((len(extant_sequence_names),len(extant_sequence_names)))

    
    phl_tree = Tree(nwk_file_name,format=1)

    for ref_num,ref_extant_sequence in enumerate(extant_sequence_names):
        for other_extant_sequence in extant_sequence_names[ref_num:]:
        
            ref_extant_sequence_t = phl_tree&ref_extant_sequence
            other_extant_sequence_t = phl_tree&other_extant_sequence
            dis_node = ref_extant_sequence_t.get_distance(other_extant_sequence_t)

            ref_idx = seq_ref_dict[ref_extant_sequence]
            other_idx = seq_ref_dict[other_extant_sequence]

            cophenetic_dist_np[ref_idx][other_idx] = dis_node
            cophenetic_dist_np[other_idx][ref_idx] = dis_node
    return cophenetic_dist_np

''' function to shuffle amino acids in the protein sequence '''
def shuff(seq,shuff_percent):
    total_aa_random = int(shuff_percent * len(seq))
    seq_list = [*seq]
    
    for r in range(0,total_aa_random):
        # swap positions
        rand_pos_i = np.random.choice(len(seq_list))
        rand_pos_j = np.random.choice(len(seq_list))
        seq_list[rand_pos_i],seq_list[rand_pos_j] = seq_list[rand_pos_j],seq_list[rand_pos_i]

    shuff_seq = ''.join(seq_list)
    return shuff_seq

def shuffle_aa(fasta_file_name_wo_gap,shuffle_percent):

    file_location  = os.path.dirname(fasta_file_name_wo_gap) 
    base_file_name = os.path.splitext(os.path.basename(fasta_file_name_wo_gap))[0]

    shuffled_fasta_file_name = f'{file_location}/{base_file_name}_{str(shuffle_percent)}.aln'

    with FastxFile(fasta_file_name_wo_gap, 'r') as fh, open(shuffled_fasta_file_name, 'w') as outfile:
      for entry in fh:
         new_shuffle_seq = shuff(entry.sequence, shuffle_percent)
         outfile.write('>' + str(entry.name) + '\n')
         outfile.write(str(new_shuffle_seq)  + '\n')
    fh.close()
    outfile.close()
    print(f"Created {shuffle_percent} shuffled sequence file")


''' function for one hot encoding '''
one_hot_dict = {"A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "C": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "D": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "E": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "F": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "G": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "H": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "I": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "K": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "L": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "M": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "N": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  "P": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
  "Q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
  "R": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
  "S": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
  "T": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
  "V": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
  "W": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
  "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
  "X": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

residue_types = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']

def mean_embed(encodings):
    x = []
    for i in range(len(encodings)):
        x.append(list(encodings[i].values())[0])
    return np.array(x).mean(axis=0)

def get_encoding(seq):
    seq = seq.upper()
    encoding_data = []
    for res in seq:
        if res not in residue_types:
            res = "X"
        encoding_data.append({res: one_hot_dict[res]})
    return encoding_data

def get_seq_oh_embedding(fasta_file_name_wo_gap):
    seq_label,oh_embedding_lst = [], []
    with FastxFile(fasta_file_name_wo_gap, 'r') as fh:
        for entry in fh:
            seq_label.append(entry.name)
            oh_encoding = get_encoding(entry.sequence)
            oh_embedding_lst.append(mean_embed(oh_encoding))

    print(f"One hot embedding array created of shape{np.array(oh_embedding_lst).shape}")
    return seq_label, np.array(oh_embedding_lst)

''' function to create euclidean distance matrix '''
def create_euclidean_distance_matrix(plm_seq_labels_dict,plm_seq_layer_embedding,uni_sequence_names_dict):

    # plm embeddings distance metric
    euc_dist_np = np.zeros((len(plm_seq_labels_dict),len(plm_seq_labels_dict)))
    # print(euc_dist_np.shape)

    plm_seq_name_list = list(plm_seq_labels_dict.keys())

    # create distance metric using embeddings using euclidean distance, cosine, TS-SS
    for ref_num, ref_extant_sequence in enumerate(plm_seq_name_list):
        for other_extant_sequence in plm_seq_name_list[ref_num + 1:]:
            
            # sequence embedding
            ref_seq_embedding   = plm_seq_layer_embedding[plm_seq_labels_dict[ref_extant_sequence]]
            other_seq_embedding = plm_seq_layer_embedding[plm_seq_labels_dict[other_extant_sequence]]

            # universal ids in the universal list
            uni_ref_idx   = uni_sequence_names_dict[ref_extant_sequence]
            uni_other_idx = uni_sequence_names_dict[other_extant_sequence]

            euc_dist = distance.euclidean(ref_seq_embedding,other_seq_embedding)
            euc_dist_np[uni_ref_idx][uni_other_idx] = euc_dist_np[uni_other_idx][uni_ref_idx] = euc_dist

    return euc_dist_np


# distance spearman rank correlation
def sperman_rank_corr(a,b): # spearman rank correlation between 2 distance matrices - in numpy array.
    # check the input type and shape
    assert type(a)==np.ndarray and type(b)==np.ndarray, "Input must be numpy array " #pd.DataFrame
    assert a.shape == b.shape, "Arrays are of different shape"

    # get the upper half of the matrix only
    mask =  np.triu_indices(a.shape[0], k=1)
    
    return stats.spearmanr(a[mask],b[mask])

def pearson_corr(a,b): # spearman rank correlation between 2 distance matrices - in numpy array.
    # check the input type and shape
    assert type(a)==np.ndarray and type(b)==np.ndarray, "Input must be numpy array " #pd.DataFrame
    assert a.shape == b.shape, "Arrays are of different shape"

    # get the upper half of the matrix only
    mask =  np.triu_indices(a.shape[0], k=1)
    
    return pearsonr(a[mask],b[mask])