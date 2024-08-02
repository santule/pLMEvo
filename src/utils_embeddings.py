'''
python script to get the embeddings for all pLMs layer wise
'''

import sys
import argparse
import torch
import os
from pathlib import Path
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from torch.utils.data import TensorDataset,Dataset
#import pandas as pd
import numpy as np
import pickle
import utils
import string, itertools
from typing import List, Tuple
from Bio import SeqIO

# pytorch dataset to save the data
class PLMEmb_Dataset(Dataset):
    def __init__(self,seq_labels,seq_embeddding):
      self.seq_labels     = seq_labels
      self.seq_embeddding = seq_embeddding
    def __len__(self):
      return len(self.seq_labels)
    def __getitem__(self, idx):
      return self.seq_labels[idx], self.seq_embeddding[idx]

def read_fasta( fasta_path ):
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case
                
    return sequences

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""    
    
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


''' function to get column attention - layer 1 head 5'''
def get_msa_colattn(total_seqs,msa_file_name,model,alphabet,device,seq_ref_dict):
  
  output_layers = [12]
  msa_batch_converter = alphabet.get_batch_converter()

  # process the MSA
  msa_data = [read_msa(msa_file_name, total_seqs)]
  msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
  print(f"PROCESS:: READ {msa_file_name} with {total_seqs} sequences")
  plm_seq_labels_dict = {label:idx for idx,label in enumerate(msa_batch_labels)}
  
  with torch.no_grad():
    out = model(msa_batch_tokens.to(device), repr_layers=output_layers, need_head_weights=False)
    col_attn = out["col_attentions"] # 1,12,12,seq_len,total_seq,total_seq
    col_attn = col_attn.cpu().numpy()[0,...].mean(axis=2) # 12,12,total_seq,total_seq

    # symmetrical column attention (addition but can be averaged?)
    col_attn += col_attn.transpose(0,1,3,2) # 12,12,total_seq,total_seq

    # rearrange col attention to match the universal matrix
    uni_col_attn = np.zeros((12,12,total_seqs,total_seqs))

    for ref_num, ref_extant_sequence in enumerate(msa_batch_labels):
        for other_extant_sequence in msa_batch_labels[ref_num + 1:]:
          ref_seq_pos   = plm_seq_labels_dict[ref_extant_sequence] # msa reference
          other_seq_pos = plm_seq_labels_dict[other_extant_sequence]

          # universal ids in the universal list
          uni_ref_idx   = seq_ref_dict[ref_extant_sequence]
          uni_other_idx = seq_ref_dict[other_extant_sequence]

          for layer in range(12):
            for head in range(12):
                uni_col_attn[layer,head,uni_ref_idx,uni_other_idx] = col_attn[layer,head,ref_seq_pos,other_seq_pos]
                uni_col_attn[layer,head,uni_other_idx,uni_ref_idx] = col_attn[layer,head,ref_seq_pos,other_seq_pos]
  
  print(f"column attention created and is of the shape {uni_col_attn.shape}")
  return uni_col_attn


def get_msa_embedding(total_seqs,msa_file_name,model,alphabet,output_layers,device):

  truncation_seq_length = 1022
  msa_batch_converter = alphabet.get_batch_converter()

  # process the MSA
  msa_data = [read_msa(msa_file_name, total_seqs)]
  msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
  print(f"PROCESS:: READ {msa_file_name} with {total_seqs} sequences")

  # list to store sequence name and embeddings for all layers
  msa_embedding = []
  
  with torch.no_grad():
    out = model(msa_batch_tokens.to(device), repr_layers=output_layers, need_head_weights=False)
    
    for layer in output_layers:
      token_representations = out["representations"][layer][:,:,1:,:] # 1st position is CLS # 1,55,275,768
      #print(f"token_representations shape {token_representations.shape}")
      mean_representation   = token_representations.mean(dim = 2).to('cpu').squeeze(0) # 55,768
      #print(f"mean_representation shape {mean_representation.shape}")
      msa_embedding.append(mean_representation.to('cpu').numpy()) # 12,55,768 # list of numpy array

  # save as pytorch dataset
  plm_embedding_tensor = torch.tensor(np.array(msa_embedding)).permute(1,0,2) # change to 55,12,768
  print(f"embeddding extracted and is of shape {plm_embedding_tensor.shape}")

  PLM_dataset = PLMEmb_Dataset(msa_batch_labels[0],plm_embedding_tensor)
  return PLM_dataset

def get_esm_embedding(fasta_file_name, model, alphabet,output_layers, device):

  tokens_per_batch   = 4096
  truncation_seq_length = 1022

  # process fasta file into pytorch dataset and dataloader
  dataset = FastaBatchedDataset.from_file(fasta_file_name) # convert into pytorch dataset
  batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
  data_loader = torch.utils.data.DataLoader(
      dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
  ) # - sequence labels, sequence, sequence tokens
  print(f"PROCESS:: READ {fasta_file_name} with {len(dataset)} sequences")

  # list to store sequence name and embeddings for all layers
  plm_embedding = []
  seq_label = []

  with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
      print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
      toks = toks.to(device=device, non_blocking=True)
      out = model(toks, repr_layers = output_layers, return_contacts = False) # embeddings from all layers

      for i, label in enumerate(labels):
        truncate_len = min(truncation_seq_length, len(strs[i]))
        seq_label.append(label)
        sequence_all_layer_embedding = [] # for each sequence
        for layer in output_layers:
          token_representations = out["representations"][layer]
          mean_representation   = token_representations[i, 1 : truncate_len + 1].mean(0).to('cpu')
          sequence_all_layer_embedding.append(mean_representation.to('cpu').numpy())

        # layer processing done for all sequences
        plm_embedding.append(sequence_all_layer_embedding)

  # save as pytorch dataset
  plm_embedding_tensor = torch.tensor(np.array(plm_embedding))
  print(f"embeddding extracted and is of shape {plm_embedding_tensor.shape}")

  # save the output in pytorch dataset and then on disk
  PLM_dataset = PLMEmb_Dataset(seq_label,plm_embedding_tensor)
  return PLM_dataset

def get_pt_embedding(fasta_file_name,model,tokenizer,output_layers,device):
  # fasta_file_wo_gap,pt_model,pt_tokenizer,output_layers,device

  plm_embedding = []
  seq_label = []
  max_seq_len  = 1000
  max_batch    = 100
  max_residues = 4000
  # Read in fasta
  seq_dict = read_fasta(fasta_file_name)
  
  avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
  n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
  seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
  
  print("Average sequence length: {}".format(avg_length))
  print("Number of sequences >{}: {}".format(max_seq_len, n_long))

  batch = list()
  for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
      seq     = seq.replace('U','X').replace('Z','X').replace('O','X')
      seq_len = len(seq)
      seq = ' '.join(list(seq))
      batch.append((pdb_id,seq,seq_len))
  
      n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
      if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
        pdb_ids, seqs, seq_lens = zip(*batch)
        batch = list()

        token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
        input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        with torch.no_grad():
          embedding_repr = model(input_ids, attention_mask=attention_mask,output_hidden_states=True)['hidden_states']
        
        # batch-size x seq_len x embedding_dim
        # extra token is added at the end of the seq
        for batch_idx, identifier in enumerate(pdb_ids):
          s_len = seq_lens[batch_idx]
          sequence_all_layer_embedding = [] # for each sequence
          for layer in output_layers:
            # slice-off padded/special tokens
            emb = embedding_repr[layer][batch_idx,:s_len]
            emb = emb.mean(dim=0).to('cpu').numpy()
            sequence_all_layer_embedding.append(emb)

          plm_embedding.append(sequence_all_layer_embedding)
          seq_label.append(identifier)
          
  print(f"Total sequences {len(seq_label)}")

  # save as pytorch dataset
  plm_embedding_tensor = torch.tensor(np.array(plm_embedding))
  print(f"plm_embedding_tensor shape {plm_embedding_tensor.shape}")
  
  # save the output in pytorch dataset and then on disk
  PLM_dataset = PLMEmb_Dataset(seq_label,plm_embedding_tensor)

  return PLM_dataset