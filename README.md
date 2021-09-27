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
      /sources/Obtain_and_save_embeddingspre_for_predefined_categories.ipynb
      ```
    * given a category, retrievel the most similar categories via calculating the cosine similarity between each categories
      ```
      similarity_given_anytopic.ipynb
      ```
3. For labelling a paper:
    * 1. get unigram, bigram and trigram in the abstract (step 2).
    * 2. save ngrams that also show up in the candidate list (step 2).
    * 3. get all nouns in the abstract (step 3).
    * 3. retrieve the related categories of nouns, and save the related categories that also show up in the candidate list (step 3).
    * 4. combine lists from step b and c (step 4).
      ```
      Label_arbitrary_paper.ipynb
      ```
      ![Workflow of labelling a given abstract](https://github.com/casszhao/FAIR/blob/main/images/how%20we%20label%20a%20abstract.jpg)
      
