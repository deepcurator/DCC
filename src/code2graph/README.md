# Code2Graph

## How to run?

The required packages are listed in [requirements.txt](requirements.txt). The following command could be used to install these packages in the target Python environment:
```
git install -r requirements.txt
```

Below are more detailed sample commands to create a new environment named 'c2g' using conda command line, to install the requirements, and to run the notebook using this new environment as its kernel:

```
conda create --name c2g
activate c2g
cd 'path-to-code2graph-folder'
git install -r requirements.txt
python -m ipykernel install --user --name=c2g
jupyter notebook
```

Select the kernel 'c2g' in the notebook, and then your notebook is ready to run. If you encounter any issues, please feel free to contact the authors.


## Notebooks

We provide two notebooks implementing the two code graph generation approaches, as detailed in our [report](reports/milestone3/).

### Computational Graph Based Approach

[notebook](testScript/computational_graph_based.ipynb)

### Light-Weight Approach

[notebook](testScript/light_weight.ipynb)

