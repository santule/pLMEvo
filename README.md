# Do protein language models learn phylogeny ?

## Description
Deep machine learning demonstrates a capacity to uncover evolutionary relationships directly from protein
sequences, in effect internalising notions inherent to classical phylogenetic tree inference. We connect these
two paradigms by assessing the capacity of protein-based language models (pLMs) to discern phylogenetic
relationships without being explicitly trained to do so. We evaluate ESM2, ProtTrans and MSA-Transformer
relative to classical phylogenetic methods, while also considering sequence insertions and deletions (indels) across
114 Pfam datasets.

## Getting Started

Running the analysis is faster on GPU. It is recommended to use a GPU with at least 45 GB of RAM or more.

### Dependencies
```
pip install -r requirements.txt

bio==1.7.1
fair-esm==2.0.0
huggingface-hub==0.24.3
pandas==2.2.2
pysam==0.22.1
scikit-learn==1.5.1
scipy==1.14.0
sentencepiece==0.2.0
torch==2.1.0
transformers==4.43.3
ete3==3.1.3
```
For FastTree 
```
conda install -c bioconda fasttree
```

### Running the investigations presented in the paper

1. Prepare data for analysis (create LG tree, LG matrix, remove and standarise gaps, and shuffles the amino acids in sequences)

<sub> ***** Note:: Aligned fasta file is needed for phylogenetic tree which constructs the distance matrix based on the tree for comparison.***** </sub>

```
python prep_data.py -a aligned fasta file

arguments:
-a aligned fasta file with full path

example:
python prep_data.py -a ../data/PF00158/PF00158.aln
```
2. One-hot correlation analysis

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python one_hot_corr.py -a aligned fasta file -m model type

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)

example:
python one_hot_corr.py -a ../data/PF00158/PF00158.aln -m esm2
```
3. Homology correlation analysis using RSS (order) / (magnitude) for low-gap and high-gap pfam datasets

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python homology_corr.py -a aligned fasta file -m model type -s shuffled fasta
                                                              -c column attention

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)
-s shuffled fasta file boolean (options: Y for shuffled)
-c column attention representation, only works for model type 'msa'. Use Y when column attention is needed
 (uses layer 1 head 5 from MSA-Transformer for this analysis)

example:
python homology_corr.py -a ../data/PF00158/PF00158.aln -m esm2 -s N -c N
or
python homology_corr.py -a ../data/PF00158/PF00158.aln -m msa -s N -c Y
```
4. Local homolog similarity analysis
   
   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python local_homolog_sim.py -a aligned fasta file -m model type -k nearest neighbours

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)
-k nearest neighbours ( 5, 10, 20)

python local_homolog_sim.py -a ../data/PF00158/PF00158.aln -m esm2 -k 5
```

5. Fine to coarse evolutionary correlation analysis

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python fine_coarse_corr.py -a aligned fasta file -m model type -c column attention

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)
-c column attention representation, only works for model type 'msa'
 (uses layer 1 head 5 from MSA-Transformer for this analysis)

example:
python fine_coarse_corr.py -a ../data/PF00158/PF00158.aln -m msa -c Y
```
6. Elastic Net regression training for salient neuron analysis

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python salient_neurons.py -a aligned fasta file -m model type

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)

example:
python salient_neurons.py -a ../data/PF00158/PF00158.aln -m esm2

```

