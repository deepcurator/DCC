# Code2graph

The code2graph is a sub-module in DCC ([Deep Code Curator](https://github.com/deepcurator/DCC)) which aims to extract sementic information from text, images, code and equation accompanied with scientific DL papers. The purpose of code2graph is to build a pipeline of methodologies to extract Resource Description Framework (RDF) graphs, particularly from the code repositories related to DL publications. The figure below illustrates the current architecture.

![](https://github.com/louisccc/DCC/blob/master/src/code2graph/figs/architecture.jpg?raw=true)

Two methodogies are studied in code2graph. 
1. The Computational Graph-Based Approach ([graphHandler.py](https://github.com/deepcurator/DCC/blob/master/src/code2graph/core/graphHandler.py))
2. The Lightweight Approach ([graphlightweight.py](https://github.com/deepcurator/DCC/blob/master/src/code2graph/core/graphlightweight.py))

You can find details from [Technical Report on Code2Graph](http://cecs.uci.edu/files/2019/05/TR-19-01.pdf). A sample visualization of the graphs generated from both methods is shown below: (using [fashion MNIST program example](https://github.com/deepcurator/DCC/blob/master/src/code2graph/test/fashion_mnist/testGraph_extensive.py))

Computational Graph-based Approach (MNist) |  The Lightweight Approach (MNist)
:-------------------------:|:-------------------------:
<img src="https://github.com/louisccc/DCC/blob/master/src/code2graph/figs/Sample_Output_0.png?raw=true">|<img src="https://github.com/louisccc/DCC/blob/master/src/code2graph/figs/Sample_Output_1_.png?raw=true" width="700">

## Software Dependencies

* `Python 3.7` or higher should be installed
* `pip` should be installed
* `git` should be installed
* `graphviz` should be installed

## Installation Guide

Step 1: Clone the git respository by running the command below.

```shell
git clone https://github.com/deepcurator/DCC.git
```

Step 2: Create  a python virtual environment using your favorite package management system (conda, virtualenv, etc).

#### via conda

```shell
conda create -n yourenvname python=3.7 anaconda     # create conda environment 
source activate yourenvname                         # activate the environment
```

Step 3: Install the required packages to your virtual environment.

#### via conda
```shell
pip install -r requirements.txt
```

## Package Dependencies
 
* jupyter==1.0.0
* jupyter-console==5.0.0
* ipython==5.3.0
* pyvis==0.1.6.0
* astor==0.7.1
* beautifulsoup4==4.7.1
* Keras==2.2.4
* matplotlib==3.0.2
* networkx==2.2
* rdflib==4.2.2
* requests==2.21.0
* scikit-learn==0.20.2
* selenium==3.141.0
* tensorflow==1.13.1
* urllib3==1.24.1
* wget==3.2
* lxml==4.3.4
* showast==0.2.4
* autopep8==1.4.4
 
## Usage Examples
### Running Computation-Based Approach
Under Construction, or you can also refer to the [notebook](testScript/computational_graph_based.ipynb).

### Running Lightweight Approach
Run the follwing command, or you can also refer to the [notebook](testScript/light_weight.ipynb).

```shell
python script_run_lightweight_method.py -ipt [PATH_TO_CODE] -opt [N [N ...]] --arg
```
-ipt: Path to directory that contains the source code of your machine learining model.

--ds: Specifies that the path (-ipt) contains a collection of repositories.

-dp: (Used along with --ds) Path to store generated outputs. 

-opt: Types of output: 1 = call graph, 2 = call tress, 3 = RDF graphs, 4 = TensorFlow sequences, 5 = RDF triples, 6 = Export RDF (turtle format).

--ct: (Used along with --ds and opt=5) Only generates combined_triples.triples.

--arg: Show arguments on graph (Hidden by default).

--url: Show url/is_type relations on graph (Hidden by default).

#### Download RDF Dataset

We ran the lightweight method on several TensorFlow papers we scraped from Paperswithcode website. You can download the RDF graphs and triples we generated [here](https://osf.io/zrusg/?view_only=f6ed10613af94c6d8050796a30f1568b).

### Running Webscraper for Paperswithcode website

```shell
python script_scrape_paperswithcode.py -cd [PATH_TO_CHROMEDRIVER]
```

-cd: Path to ChromeDriver. To get the ChromeDriver compatible with your browser go to the following website - http://chromedriver.chromium.org/downloads and download the ChromeDriver for the version of Chrome you are using. 

### Running The Summary File Extractor
### Running Computation-Based Approach
