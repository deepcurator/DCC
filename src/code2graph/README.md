# Code2graph

The code2graph is a Python module that aims to transform the source code related to Deep Learning Architectures and methodologies into RDF graphs. In code2graph, the building blocks of the pipeline are implemented with a flexible architecture. 
The code2graph is a module in project [Deep Code Curator](https://github.com/deepcurator/DCC) (DCC) which aims to extract the information from scientific publications and the corresponding source code related to Deep Learning architectures and methodologies.

Currently, two methodogies are included in code2graph. 
1. Computation-based Approach, see [graphHandler.py](https://github.uci.edu/AICPS/code2graph/blob/master/core/graphHandler.py).
2. The Lightweight Approach, see [graphlightweight.py](https://github.uci.edu/AICPS/code2graph/blob/master/core/graphlightweight.py).

Computation-based Approach (MNist) |  The Lightweight Approach (VGG)
:-------------------------:|:-------------------------:
![](https://github.uci.edu/AICPS/code2graph/blob/master/figs/Sample_Output_0.png?raw=true) | ![](https://github.uci.edu/AICPS/code2graph/blob/master/figs/Sample_Output_1_.png?raw=true)

The following figure illustrates the current pipeline architecture of code2graph:
![](https://github.uci.edu/AICPS/code2graph/blob/master/figs/architecture.jpg?raw=true)

To understand the pipeline of code2graph better, you can refer to 
- [Deep Code Curator - Technical Report on Code2Graph](http://cecs.uci.edu/files/2019/05/TR-19-01.pdf)

## Software Dependencies

* `Python 3.7` or higher should be installed
* `pip` should be installed
* `git` should be installed
* `graphviz` should be installed

## Installation Guide

Step 1: Clone the git respository by running one of the commands shown in the following snippets.

```shell
git clone https://github.uci.edu/AICPS/code2graph.git
git clone git@github.uci.edu:AICPS/code2graph.git
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
## Usage Examples
### Running Computation-Based Approach
Under Construction, or you can also refer to the [notebook](testScript/computational_graph_based.ipynb).

### Running Lightweight Approach
Run the follwing command, or you can also refer to the [notebook](testScript/light_weight.ipynb).

```shell
python script_run_lightweight_method.py -ipt [PATH_TO_CODE] -opt [N [N ...]] --arg
```
-ipt: Path to directory that contains the source code of your machine learining model.

-opt: Types of output: 1 = call graph, 2 = call tress, 3 = RDF graphs, 4 = TensorFlow sequences.

--arg: Show arguments on graph (Hidden by default).

--url: Show url/is_type relations on graph (Hidden by default).

### Running Webscraper for Paperswithcode website

```shell
python script_scrape_paperswithcode.py -cd [PATH_TO_CHROMEDRIVER]
```

-cd: Path to ChromeDriver. To get the ChromeDriver compatible with your browser go to the following website - http://chromedriver.chromium.org/downloads and download the ChromeDriver for the version of Chrome you are using. 

### Running The Summary File Extractor
### Running Computation-Based Approach
