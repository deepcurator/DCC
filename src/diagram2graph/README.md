# Image2Graph

## How to run?

The required packages are listed in [requirements.txt](requirements.txt). The following command could be used to install these packages in the target Python environment:

```
git install -r requirements.txt
```

### OS Based Differences

<b>For Windows:</b> 

Tesseract needs to be installed using the installer from the following link: https://github.com/UB-Mannheim/tesseract/wiki
Please make sure that the following line of code is uncommented in the notebook, and has the correct path to the tesseract executable in your local.

```
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

<b>For Linux based systems:</b> 

Please make sure that the above line is commented in the notebook.

### Detailed Instructions

Below are more detailed sample commands to create a new environment named 'i2g' using conda command line, to install the requirements, and to run the notebook using this new environment as its kernel:

```
conda create --name i2g
activate i2g
cd 'path-to-image2graph-folder'
git install -r requirements.txt
python -m ipykernel install --user --name=i2g
jupyter notebook
```

Select the kernel 'i2g' in the notebook, and then your notebook is ready to run. If you encounter any issues, please feel free to contact the authors.
