# ProtCNN
Reproduction of ProtCNN according to [Using Deep Learning to Annotate the Protein Universe](https://www.biorxiv.org/content/10.1101/626507v2.full).

PFAM *seed* data avaliable at [here](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split).

process data with `process_data.ipynb` based on [here](https://github.com/anindya-vedant/Genetic-ProtCNN/blob/master/Notebooks/Pfam_protein_sequence_classification_full.ipynb)

pad all the sequences to length of 2048
train with
```shell
python run_2048.py
```
