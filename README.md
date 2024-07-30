# Do protein language model learn phylogenetic relationships?


To run the investigations presented in the paper :

1. Prepare data for analysis (create LG tree, LG matrix, remove and standarise gaps, and shuffles the amino acids in sequences)
```
python prep_data.py -aln aligned_fasta_file

arguments:
-aln aligned fasta file with full path

example:
python prep_data.py -aln ../data/PF01196/PF01196.aln

```
2. One-hot analysis
```
python run_one_hot_analysis.py --path --model 

```
3. RSS (order) / (magnitude) to low-gap and high-gap pfam datasets
```
python rss_analysis.py --path --model

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
