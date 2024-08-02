''' python script to wrapper around the embeddings script '''

import os
import utils_embeddings as utils_embeddings
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from transformers import T5Tokenizer, T5EncoderModel
import torch, sys
import pickle, csv
import utils, os
from pysam import FastaFile,FastxFile
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
total_layers_model = {'msa':13,'esm2':49,'pt':25}
full_model_name = {'msa':'esm_msa1b_t12_100M_UR50S','esm2':'esm2_t48_15B_UR50D','pt':'Rostlab/prot_t5_xxl_uniref50'}

def check_msa_stats(msa_file):
  total_seqs = 0
  with FastxFile(msa_file, 'r') as fh:
    for entry in fh:
      total_seqs += 1
      size_align_seq = len(entry.sequence)
  return total_seqs,size_align_seq

def get_col_attn(fasta_file_std_gap,seq_ref_dict):
    model_name = 'esm_msa1b_t12_100M_UR50S'
    msa_model, msa_alphabet = pretrained.load_model_and_alphabet(model_name)
    msa_model.to(device)
    msa_model.eval()

    # check total sequences and if aligned seq < 1024
    total_seqs,size_align_seq = check_msa_stats(fasta_file_std_gap)
    print(f"Total sequences is {total_seqs} and sequence length is {size_align_seq}")
    if size_align_seq > 1024 or total_seqs > 1024:
        print("Sequence greater than 1024, sequence would be truncated. So Skipping.")
        return
    
    utils_embeddings.get_msa_colattn(total_seqs,fasta_file_std_gap,msa_model,msa_alphabet,device,seq_ref_dict)



def get_plm_representation(model_type,fasta_file_wo_gap,fasta_file_std_gap,layers):

    if model_type == 'esm2':
        model_name = 'esm2_t48_15B_UR50D'
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet(model_name)
        esm_model.to(device)
        esm_model.eval()
        if layers == 'all':
           output_layers = list(range(0,total_layers_model[model_type]))
        else:
            output_layers = layers
        PLM_dataset = utils_embeddings.get_esm_embedding(fasta_file_wo_gap,esm_model,esm_alphabet,output_layers,device)


    if model_type == 'msa':
        model_name = 'esm_msa1b_t12_100M_UR50S'
        msa_model, msa_alphabet = pretrained.load_model_and_alphabet(model_name)
        msa_model.to(device)
        msa_model.eval()
        if layers == 'all':
           output_layers = list(range(0,total_layers_model[model_type]))
        else:
            output_layers = layers
        # check total sequences and if aligned seq < 1024
        total_seqs,size_align_seq = check_msa_stats(fasta_file_std_gap)
        print(f"Total sequences is {total_seqs} and sequence length is {size_align_seq}")
        if size_align_seq > 1024 or total_seqs > 1024:
            print("Sequence greater than 1024, sequence would be truncated. So Skipping.")
            return
        PLM_dataset = utils_embeddings.get_msa_embedding(total_seqs,fasta_file_std_gap,msa_model,msa_alphabet,output_layers,device)

    if model_type == 'pt':
        model_name = 'Rostlab/prot_t5_xxl_uniref50'
        pt_tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        pt_model = T5EncoderModel.from_pretrained(model_name)
        pt_model.to(device)
        pt_model.eval()
        if layers == 'all':
           output_layers = list(range(0,total_layers_model[model_type]))
        else:
            output_layers = layers
        PLM_dataset = utils_embeddings.get_pt_embedding(fasta_file_wo_gap,pt_model,pt_tokenizer,output_layers,device)
    
    return PLM_dataset
                



