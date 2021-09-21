# FAIR:Finding Accessible Inequalities Research in Public Health (the FAIR Database)

## Requirements
```shell script
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext
# requirements from pytorch-transformers/wiki
pip install transformers pymediawiki
```
## Workflow
1. Get pre-defined wikipedia categories (we call it candidate categories/candidate list). These categories are the ones we want to use to summarize/label a given abstract/paper (We also mannually reviewed the list and removed categories that are not relavent).
2. For finding similar and related topics:
    * get a ClinicalBERT embeddings for each categories (in the candidate categories)
      ```
      Obtain_and_save_pre_defined_categories_embeddings.ipynb
      ```
    * given a category, retrievel the most similar categories via calculating the cosine similarity between each categories
      ```
      similarity_given_anytopic.ipynb
      ```
3. For labelling a paper:
    * 1. filter out all nouns
    * 2. save nounts that also show up in the candidate list
    * 3. retrieve the related categories of nouns, and keep the ones also in the candidate list
    * 4. combine lists from step b and c.
      ```
      Label_arbitrary_paper.ipynb
      ```
