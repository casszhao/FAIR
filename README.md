# FAIR:Finding Accessible Inequalities Research in Public Health (the FAIR Database)

## Requirements
```shell script
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext
# requirements from pytorch-transformers/wiki
pip install transformers pymediawiki
```
## Workflow
1. Get pre-defined wikipedia categories (we call it candidate categories). These categories are the ones we want to choose from to summarize/label a given abstract/paper.
2. For finding similar and related topics:
    * get a ClinicalBERT embeddings for each categories (in the )
    * calculate the cosine similarity between each categories
    * given a category, retrievel the most cosine similar categories 
3. For labelling a paper:
    * 
