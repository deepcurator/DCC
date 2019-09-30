# Code2graph

The code2graph is a sub-module in DCC ([Deep Code Curator](https://github.com/deepcurator/DCC)) which aims to extract sementic information from text, images, code and equation accompanied with scientific DL papers. The purpose of code2graph is to build a pipeline of methodologies to extract Resource Description Framework (RDF) graphs, particularly from the code repositories related to DL publications. The figure below illustrates the current architecture.

![](https://github.com/louisccc/DCC/blob/master/src/code2graph/figs/architecture.jpg?raw=true)

Two methodogies are studied in code2graph. 
1. The Computational Graph-Based Approach ([graphHandler.py](https://github.com/deepcurator/DCC/blob/master/src/code2graph/core/graphHandler.py))
2. The Lightweight Approach ([graphlightweight.py](https://github.com/deepcurator/DCC/blob/master/src/code2graph/core/graphlightweight.py))

You can find details from [Technical Report on Code2Graph](http://cecs.uci.edu/files/2019/05/TR-19-01.pdf). A sample visualization of the graphs generated from both methods is shown below: (using [fashion MNIST program example](https://github.com/deepcurator/DCC/blob/master/src/code2graph/test/fashion_mnist/testGraph_extensive.py))

Computational Graph-based Approach (MNist) |  The Lightweight Approach (MNist)
:-------------------------:|:-------------------------:
<img src="https://github.com/louisccc/DCC/blob/master/src/code2graph/figs/Sample_Output_0.png?raw=true">|<img src="https://github.com/louisccc/DCC/blob/master/src/code2graph/figs/Sample_Output_1_.png?raw=true" width="850">

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

* jupyter==1.0.0 => Jupyter notebook.
* jupyter-console==5.0.0 => Jupyter notebook.
* ipython==5.3.0 => Jupyter notebook.
* pyvis==0.1.6.0 => RDF graph visualization.
* astor==0.7.1  => AST manipulation and printing.
* beautifulsoup4==4.7.1 => Webscraping.
* Keras==2.2.4 => Compile Keras projects.
* tensorflow==1.13.1 => Compile tensorflow projects.
* matplotlib==3.0.2 
* networkx==2.2
* rdflib==4.2.2 => RDF graph construction.
* requests==2.21.0 => Webscraping.
* scikit-learn==0.20.2
* selenium==3.141.0 => Webscraping.
* urllib3==1.24.1 => Webscraping.
* wget==3.2 => Webscraping.
* lxml==4.3.4 => Webscraping.
* showast==0.2.4 => Visualizing AST.
* autopep8==1.4.4 => Preprocess data.
* apscheduler==3.6.1 => Scheduler for web crawler.
 
## Usage Examples
### Running Computation-Based Approach
Refer to the [notebook](testScript/computational_graph_based.ipynb).

### Running Lightweight Approach
Refer to the [notebook](testScript/light_weight.ipynb).

## Dataset

Using our script we scraped around 600 papers from paperswithcode.com website. Out of 600 papers, 120 of them have tensorflow implementation. We ran the lightweight method on those TensorFlow papers we scraped from Paperswithcode website. The lightweight method was successful on half of the tensorflow repositories. You can download the RDF graphs and triples we generated [here](https://osf.io/zrusg/?view_only=f6ed10613af94c6d8050796a30f1568b).

### Running Webscraper for Paperswithcode website

```shell
python script_service_pwc_scraper.py -cd [PATH_TO_CHROMEDRIVER] -sp [SAVE_PATH]
```

-cd: Path to ChromeDriver. To get the ChromeDriver compatible with your browser go to the following website - [ChromeDriver](http://chromedriver.chromium.org/downloads) and download the ChromeDriver for the version of Chrome you are using.

-sp: The script will save the scraped data in this path.
