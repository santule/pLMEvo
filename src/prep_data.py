'''
script to remove and standarise gaps, 
generate LG Tree, 
generate LG matrix,
shuffle amino acids in aligned and unaligned fasta file.
'''
import sys, utils, pickle, os
import numpy as np
import getopt

# help function
def help():
    print("Incorrect or Incomplete command line arguments")
    print('python prep_data.py -a alignment file')
    exit()


def prep_data_fn(fasta_aln_file):

    file_location  = os.path.dirname(fasta_aln_file) 
    base_file_name = os.path.splitext(os.path.basename(fasta_aln_file))[0]

    # check for illegal AA
    check_aa = utils.aa_check(fasta_aln_file)
    if not check_aa:
        print(f"AA check not passed. Illegal characters found. Would not be processing Further.")
        return
    
    # convert to std gap characters
    fasta_file_name_std_gap = f'{file_location}/{base_file_name}_stdgap.aln'
    utils.std_gap_chars(fasta_aln_file,fasta_file_name_std_gap,'.')

    # remove gaps
    fasta_file_name_wo_gap = f'{file_location}/{base_file_name}_nogap.aln'
    utils.remove_gaps(fasta_aln_file,fasta_file_name_wo_gap,'.')
    print(f"Create fasta file without gaps")
    
    # shuffle aa in both gap and non-gap fasta file
    print(f"Creating shuffled aa file for no gap fasta file.")
    utils.shuffle_aa(fasta_file_name_wo_gap,0.8)
    print(f"Creating shuffled aa file for std gap fasta file.")
    utils.shuffle_aa(fasta_file_name_std_gap,0.8)
   
    # create a reference sequence dictionary
    seq_ref_dict = utils.create_ref_idx(fasta_aln_file)
    with open(f'{file_location}/sequence_ref_dict.pkl', 'wb') as fp:
        pickle.dump(seq_ref_dict, fp) 
    print(f"Created sequence reference dictionary")

    # generate tree
    nwk_file_name = f'{file_location}/{base_file_name}.tree'
    utils.create_lg_tree(fasta_file_name_std_gap,nwk_file_name)

    # generate LG matrix
    lg_mat_np = utils.generate_lg_matrix(seq_ref_dict,nwk_file_name)
    np.save(f'{file_location}/lg_mat.npy',lg_mat_np)
    print(f"Created LG matrix")
    
if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, _ = getopt.getopt(argv, "a:")

    if len(opts) == 0:
        help()
    
    for opt, arg in opts:
        if opt in ['-a']:
            fasta_aln_file = arg

    prep_data_fn(fasta_aln_file)