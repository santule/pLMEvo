# Do protein language model learn phylogenetic relationships?


To run the investigations presented in the paper :

1. Prepare data for analysis (create LG tree, LG matrix, remove and standarise gaps, and shuffles the amino acids in sequences)
```
python prep_data.py -a aligned fasta file

arguments:
-a aligned fasta file with full path

example:
python prep_data.py -a ../data/PF01196/PF01196.aln

```
2. One-hot analysis
```
python run_one_hot_analysis.py -a aligned fasta file -m model type

arguments:
-a aligned fasta file with full path
-m model type (options: esm2, pt, msa)

example:
python run_one_hot_analysis.py -a ../data/PF01196/PF01196.aln -m esm2
```
3. Homology analysis using RSS (order) / (magnitude) for low-gap and high-gap pfam datasets
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
```
python evol_timescale_analysis.py --path --model 

```
5. Elastic Net Regression Training and salient neuron analysis
```
python salient_neurons.py --path --model

```
6. Non-homologous dispersion probe
```
python nonhomologous_probe.py --path --otherpath --model

```
