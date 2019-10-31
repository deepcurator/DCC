# DCC Queries Demo Installation

In this page we provide instructions on the installation of the Deep Code Curator Queries [demo](Queries.ipynb).

## Setup Python Environment

You may use [dcc.yml](../../dcc.yml) to create the Python environment following the below steps. **If you have already setup this environment for the [other DCC demo notebook](Deep%20Code%20Curator%20(DCC).ipynb), feel free to skip this step.**
- Run `conda env create -f dcc.yml` in a **Windows Command** window. The reason we recommend to use Windows Command window is that some versions of Conda Command window has a bug installing 'pip' requirements. If you get the error 'conda not found', add the following lines (or corresponding locations from your computer) into your 'path' system variable: `C:\Users\YOUR_USERNAME\AppData\Local\Continuum\anaconda3`, `C:\Users\YOUR_USERNAME\AppData\Local\Continuum\anaconda3\Scripts`.
- Switch to Anaconda Command window and activate the new environment: `activate dcc`
- Add your new environment to Python ipykernel by running the following command `python -m ipykernel install --user --name=dcc`
- Try running jupyter notebook using: `jupyter-notebook`.

- Note: If you run into a "DLL not found" error during the above steps, run the following commands: `pip uninstall pyzmq`, `pip install pyzmq` and then try the step again.

## Install Virtuoso
Virtuoso is the database used to store the graph data for DCC. Below we provide a summary of the installation steps. For detailed instructions, please check [the corresponding Virtuoso page](http://vos.openlinksw.com/owiki/wiki/VOS/VOSUsageWindows).
- Pre-built binaries of Virtuosofor Windows require the Microsoft Visual C++ 2012 Redistributable Package. If you do not already have it in your system, it can be downloaded from [the corresponding Microsoft page](https://www.microsoft.com/en-us/download/details.aspx?id=30679#).
- Next, [download](https://sourceforge.net/projects/virtuoso/files/virtuoso/7.2.5/Virtuoso_OpenSource_Server_7.20.x64.exe/download) and install the pre-built Virtuoso (for Windows).
- Setup the necessary environment variables. Determine the location for your Virtuoso installation (e.g. `C:/Program Files/OpenLink Software/VOS7/virtuoso-opensource/`). Using this path, create a new system environment variable called `VIRTUOSO_HOME`.
- Finally, add the following string to the end of the existing PATH system variable: `;%VIRTUOSO_HOME%/bin;%VIRTUOSO_HOME%/lib`.


## Download and Import Data
We provide sthe data file `consolidated.ttl` through the `demo_queries` folder in the corresponding [OSF project](https://osf.io/jdhw8/). After you download this file into your computer, following steps will guide you to import it to the Virtuoso:
- Open http://localhost:8890 in your Web browser. If you do not use your localhost for Virtuoso, or if you use another port, update the address accordingly.
- Enter account username `dba` and password `dba` to login.
- Navigate to the window Linked Data → Quad Store Upload
- Choose the downloaded `consolidated.ttl` file to upload, and specify the Named Graph IRI as `https://github.com/deepcurator/DCC/`. Then click 'Upload".
- After seeing the upload completed message, click on the "Graphs -> Graphs" tabs to verify the existence of the uploaded database. You would be seeing the new named graph (`https://github.com/deepcurator/DCC/`) in that list.

Now you can start using the Queries [demo](Queries.ipynb). You will need to update your Virtuoso address in the notebook if you use something other than the default `http://localhost:8890`.


# Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990010
 
