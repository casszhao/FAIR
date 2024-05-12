Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg



# FAIR:Finding Accessible Inequalities Research in Public Health (the FAIR Database)
This is the source code repo for project [FAIR](https://eppi.ioe.ac.uk/EPPI-Vis/Fair)





System Overview
 ![System overview](https://github.com/casszhao/FAIR/blob/main/images/SystemOverview.jpg)

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
      

4. PPlus_classifier contains two models for PROGRESS-Plus classifiers.
