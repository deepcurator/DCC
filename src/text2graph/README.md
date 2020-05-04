
# Text2Graph

## Introduction

Here we focus on curation of text from scientific publications. The process consists of multiple steps described below (configuration file is located in {root}\conf\conf.yaml):
- We start with the data (in the form of pdf files) collected in {RAW_DATA_FOLDER}
- Script [extract_text_all_papers.py](extract_text_all_papers.py) extracts text from the pdfs and stores it as .txt files in {EXTRACT_TEXT_PATH}
- Script [extract_abstract_from_text.py](extract_abstract_from_text.py)  extracts abstracts from text file, and stores them in (numbered) text files in {EXTRACT_ABSTRACT_PATH} 
- At this point the training data can be annotated manually (using BRAT), or using some annotation techniques described below. However they are obtained, the annotations are stored in {ANNOTATED_TEXT_PATH}. 
- For some analysis, annotations have to be separated into sentence level. This is accomplished by script [break_brat.py](break_brat.py) (results can be stored in {SENTENCE_ANNOTATED_TEXT_PATH}. 
- Now, Named Entity Recognition (NER) and Relation Extraction (RE) models can be trained and applied:
	- Script [train_dcc_entities.py] puts together all pieces of NER
	- BRAT annotations can be converted to SPACY format with [brat2spacy.py]
	- File [ner_utils.py] contain helpful functions such as ner_eval and test_ner_model
	- Script [models.py] does RE model learning. 

## Examples:

We provide Jupyter notebooks to showcase the main functionalities:

- [Train NER](train_dcc_entities.ipynb) - NER model training and validation pipeline (generated models used in next notebook)
- [Text2Graph](text2graph.ipynb) - overall text2graph workflow

## Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990010
 

