
# Deep Code Curation (DCC)

## Introduction

Artificial Intelligence (AI) techniques play a central role in any data-driven application. This growth has been spurned by the advances in the field of Deep Learning (DL). In all major Machine Learning (ML) conferences, the largest percentage of submitted and accepted papers deal with new DL models, architectures, or algorithmic approaches. Most of these papers are accompanied by source code which is publicly available. The growth is expected to increase in the future as DL keeps making major advances in medicine (e.g., by predicting the effect of mutations in non-coding DNA), and in Natural Language Processing (e.g., language translation). Keeping up with this large growth of DL publications and source code has been a challenge for the researchers and practitioners. Fortunately, the DL community has some established guidelines that try to manage this growth. More specifically the DL community has:

* **Standardized frameworks:**  Most DL frameworks are written on standard tools such as Tensorflow, Keras, PyTorch, etc. These frameworks are well documented and have an active community. Being able to rapidly prototype scientific results is a must for any community.

* **Reproducibility:** The deep learning community that oversees modern conferences such as NIPS, ICML, ICLR, CVPR and KDD have emphasized the need for publications to accompany code that is reproducible and aligned with the results. This serves as a good benchmark for any scientific field.

* **Open review system:**  Publication processes in the deep learning community also follow an open review system, thereby having a more thorough feedback of scientific advancements. This transparency in the field of scientific learning makes it ideal for machine curation and validation.

These standards seek to establish a common framework for the dissemination of scientific information. Our system, **Deep Code Curator (DCC)**, takes advantage of these processes and seeks to unify them by providing a common machine queryable representation of each scientific paper with all its major modalities: text, image, and source code. This will allow scientists and engineers to explore the vast knowledge that resides in these papers and identify atomic scientific facts and their implementations. For example a scientist from a different field (e.g., psychology) interested in learning and implementing a Recurrent Neural Network (which is a specific DL model) can issue the following query (which currently cannot be done): *“Find all papers with a Bi-Directional Recurrent Neural Network model, implanted in Python and TensorFlow, in the area of Natural Language Processing, published between 2010 and 2018, in conferences like NIPS, ICML, ACL and EMNLP.”*

The impact of the Deep Code Curator is that it will dramatically decrease the time, effort and cost spent in curating the deep learning literature and associated source code. This will drive faster adoption of AI techniques in products and services that are vital to national security and industrial and scientific innovation. 

## Architecture

The overall architecture of our system is shown in the following figure.

<p align="center">
 <img align="center" src="Picture1.png" alt="generalarchitecture">
</p>

The system consists of two main parts:

The first part (top part of the figure) shows the details for the proposed curation process, where publications describing deep learning methods along with their source codes are sent to a graph extraction module. This module is able to extract atomic facts from the three major modalities: text, images, and code. These facts can be quite diverse and contain overlapping and complementary information. To convert this to a standard Knowledge Graph representation model, they are sent to a mapping system, that converts these facts to RDF triples with the right vocabularies. Next the knowledge graphs are aligned based on their structural similarities. The final aligned graph is then stored in a data store known as *triple stores*. These triple stores can be queried by pattern query languages (e.g. SPARQL), that would allow expressive querying with complex conditions. 

In the second component (at the bottom of the overall architecture) we show how the curated knowledge graphs can be used to automatically infer code for papers that are not accompanied by implementation details (source code). First an aligned knowledge graph, without the components associated with the source code is extracted from those papers. Next the corresponding code graph is inferred by using previous related papers that are accompanied with code. More details can be found in the "Reports" folder.

## Organization 

The code developed in this projet is stored in the current repository. The structure is:
- conf: folder containing conf.yaml - configuration file specifying paths to different folders, enabling the scripts to function without hardcoding paths
- demo: folder containing high-level demos (descibed in the next section)
- reports: reprets presented to DARPA at project milestones
- src: folder containing code implementing the actual functionality, including:
	- alignment: combining human annotations with CSO ontology
	- code2graph
	- diagram2graph
	- embedding: code for graph embeddings and their visualizations. The input data for this can be found in OSF: [https://osf.io/tks79/](https://osf.io/tks79/)
	- kgconstruction: code for processing and combining outputs of individual modalities 
	- scrapers: code used to scrape some of Papers With Code 
	- text2graph
- dcc.yaml: environment file for working with the project
- The following folders are not included in the repository, but we recommend users create them to match conf.yaml and for better organization. Their contents can be obtained at project's [OSF repository](https://osf.io/jdhw8/):
	- Data: 
	  - A spreadhseet with information on papers we used, as well as pdfs of these papers are at [pwc_edited_plt.zip](https://osf.io/7qt8h/)
	  - The graph embedding files together with metadata (see [src/embeddings](src/embeddings) for details on their generation): [embedding_results.zip](https://osf.io/cnwfz/)
	- Models: the trained models and data structures - [Models.zip](https://osf.io/xd7rz/)
	- Ontology: the ontology structure can be found in [Ontology.zip](https://osf.io/bq5h6/)
	- Output: 


## Demos 

We provide several Jupyter notebooks to run main functionalities and to analyze results:

- [The first notebook](demo/run_all_modalities/Paper2Graph.ipynb) allows running all three pipelines (text2graph, image2graph, code2graph) on a specified paper and its corresponding source code, and the generates output RDF graph files for each modality and for the combination.

- [The second notebook](demo/run_queries/Queries.ipynb) allows running SPARQL queries on a RDF dataset generated using hundreds of papers and their corresponding source code repositories.

- [The third notebook](demo/embedding_demo/EmbeddingDemo.ipynb) demonstrates visualizations of graph embeddings for individual modalities and in combination 

There are also demos in the individual modalities: code2graph, diagram2graph, text2graph and in embedding. The README files accompanying these demos also describe the necessary setup.

## Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990010
 

