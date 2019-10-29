# Deep Code Curator Demo Installation

We provide instructions on the installation of the demo requirements.

## Python Environment

You may use [dcc.yml](dcc.yml) to create the Python environment following the below steps:
- Run `conda env create -f dcc.yml` in a **Windows Command** window.
- Switch to Anaconda Command window and activate the new environment: `activate dcc`
- Add your new environment to Python ipykernel by running the following command `python -m ipykernel install --user --name=dcc`
- Try running jupyter notebook using: `jupyter-notebook`. If you run into a "DLL not found" error, run the following commands: `pip uninstall pyzmq`, `pip install pyzmq` and then start the notebook again.

## Grobid
[Grobid](https://github.com/kermitt2/grobid) is used by the text2graph module to extract text from PDF files.

### Grobid Server
Grobid needs Gradle to be built, and based on our tests we found Gradle 4.10 and Grobid 0.5.5 works smoothly together, and hence we recommend using these versions.

- Install [gradle-4.10](https://gradle.org/next-steps/?version=4.10&format=bin) using the instructions [here](https://docs.gradle.org/current/userguide/installation.html)
- Download and extract [grobid-0.5.5](https://github.com/kermitt2/grobid/archive/0.5.5.zip)
- cd into grobid-0.5.5, and run the command "gradle clean install"
- At this point, Gradle server is built. Before running the demo, make sure to run "gradle run" in the grobid-0.5.5 folder to start the server.

### Grobid Client
Download [Grobid client](https://github.com/kermitt2/grobid-client-python) and extract it to the demo folder (dcc\demo\grobid-client-python).

## Tesseract
[Tesseract](https://github.com/tesseract-ocr/tesseract) is used by the image2graph module.

For Windows only: Tesseract needs to be installed using the installer from the following link: https://github.com/UB-Mannheim/tesseract/wiki 
Please make sure that the following line of code is uncommented in the notebook, and has the correct path to the tesseract executable in your local.

```
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
## Other Files
We provide some additional files through the [OSF project](https://osf.io/jdhw8/) of DCC.

### PDFFigures 2.0
[PDFFigures 2.0](https://github.com/allenai/pdffigures2) is used by the image2graph module to extract the images from pdf files. To make its installation more convenient, we provide compiled jar files in the zip named pdffigures.zip. These jar files need to be extracted into the dcc/demo folder.

### Models
Model files for text2graph (text2graph_models.zip) and image2graph (image2graph_models.zip) modules are provided. These zips need to be extracted into seperate folders, and their paths need to be updated in the demo notebook.


# Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990010
 
