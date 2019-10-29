
# Deep Code Curator Demo Installation

## python Environment

Use dcc.yml](dcc.yml) 


## Other dependencies

### Grobid Server

We have tested it using gradle version 4.10 and Grobid version 0.5.5. Below instructions make use of 
Needed by text2graph

- Install gradle-4.10 [https://gradle.org/next-steps/?version=4.10&format=bin] using the instructions: https://docs.gradle.org/current/userguide/installation.html
- Download and extract grobid-0.5.5 [https://github.com/kermitt2/grobid/archive/0.5.5.zip]
- "gradle clean install"

To run the server
- "gradle run"

### Grobid Client
Needed by text2graph
Download Grobid client from https://github.com/kermitt2/grobid-client-python and extract it to the demo folder. The resulting folder structure will be: demo\grobid-client-python

### Tesseract
Needed by image2graph
For Windows only: Tesseract needs to be installed using the installer from the following link: https://github.com/UB-Mannheim/tesseract/wiki Please make sure that the following line of code is uncommented in the notebook, and has the correct path to the tesseract executable in your local.

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

### pdffigures files
Needed by image2graph

### Models



## Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990010
 

