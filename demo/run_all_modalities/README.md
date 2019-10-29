# Deep Code Curator Demo Installation

## python Environment

Use dcc.yml](dcc.yml) 

## Grobid
[Grobid](https://github.com/kermitt2/grobid) is used by the text2graph module to extract text from PDF files.

### Grobid Server
Grobid needs Gradle to be built, and based on our tests we found Gradle 4.10 and Grobid 0.5.5 works smoothly together, and hence recommend using these versions.

- Install [gradle-4.10](https://gradle.org/next-steps/?version=4.10&format=bin) using the instructions [here](https://docs.gradle.org/current/userguide/installation.html)
- Download and extract [grobid-0.5.5](https://github.com/kermitt2/grobid/archive/0.5.5.zip)
- cd into grobid-0.5.5, and run the command "gradle clean install"
- At this point, Gradle server is built. Before running the demo, make sure to run "gradle run" in the grobid-0.5.5 folder to start the server.

### Grobid Client
Download [Grobid client](https://github.com/kermitt2/grobid-client-python) and extract it to the demo folder (demo\grobid-client-python).

## Tesseract
[Tesseract](https://github.com/tesseract-ocr/tesseract) is used by the image2graph module.

For Windows only: Tesseract needs to be installed using the installer from the following link: https://github.com/UB-Mannheim/tesseract/wiki 
Please make sure that the following line of code is uncommented in the notebook, and has the correct path to the tesseract executable in your local.

```
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## pdffigures files
Needed by image2graph

## Models



# Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990010
 
