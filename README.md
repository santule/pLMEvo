# Do protein language model learn phylogenetic relationships?

## Description

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

### Running the investigations presented in the paper

1. Prepare data for analysis (create LG tree, LG matrix, remove and standarise gaps, and shuffles the amino acids in sequences)

<sub> ***** Note:: Aligned fasta file is needed for phylogenetic tree which constructs the distance matrix based on the tree for comparison.***** </sub>

```
python prep_data.py -a aligned fasta file

arguments:
-a aligned fasta file with full path

example:
python prep_data.py -a ../data/PF01196/PF01196.aln
```
2. One-hot analysis

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python run_one_hot_analysis.py -a aligned fasta file -m model type

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)

example:
python run_one_hot_analysis.py -a ../data/PF01196/PF01196.aln -m esm2
```
3. Homology analysis using RSS (order) / (magnitude) for low-gap and high-gap pfam datasets

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python homology_analysis.py -a aligned fasta file -m model type -s shuffled fasta

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)
-s shuffled fasta file boolean (options: Y for shuffled)

example:
python homology_analysis.py -a ../data/PF01196/PF01196.aln -m esm2 -s N
```
4. Fine to coarse evolutionary timescale analysis

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python evol_timescale_analysis.py -a aligned fasta file -m model type

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)

example:
python evol_timescale_analysis.py -a ../data/PF01196/PF01196.aln -m esm2
```
5. Elastic Net Regression Training and salient neuron analysis

   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python salient_neurons.py -a aligned fasta file -m model type

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)

example:
python salient_neurons.py -a ../data/PF01196/PF01196.aln -m esm2

```
6. Non-homologous dispersion probe
   
   <sub> ***** Note:: Run step 1 first to ensure all files are created. ***** </sub>
```
python nonhomologous_probe.py --path --otherpath --model

```
