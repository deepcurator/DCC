# Image2Graph

This folder contains code for image2graph, our end-to-end framework that automatically locates all figures in a research paper, classifies them, extracts the content of the DL architecture figures and represents them in a form of a graph. We use [PDFFigures2](https://github.com/allenai/pdffigures2) as the tool to autoamtically extract a set of figures from a research paper. We have downloaded 1000 papers from [arXiv.org](arxiv.org) using "deep learning" as the input query. A total of 8310 figures were extracted from those papers and used as training datasets for the learning algorithms. These datasets can be found at our Open Science Framework data repository  [DCC](https://osf.io/jdhw8/?view_only=f6ed10613af94c6d8050796a30f1568b)

The following tasks can be performed by the current implementation of the image2graph module:

- **Task 1:** Perform binary classification. The aim is to classify the extracted figures into two classes, namely, diagrams dipicting DL architectures, and all others diagrams (for example, diagrams showing plots, tables, photos, etc.). The relevant code can be found in the folder: “diagram2graph\FigClassify\BinaryClassify”.
	- Step 1: Extract features. Requirement: The dataset "DATASET_Binary_V1.0". Code to run: “compute_features.py”. Command to Run: python compute_features.py --model <model name (vgg16, vgg19, resnet)> --datasetdir --outputdir
	- Step 2: Train classifiers. Requirements: The dataset "DATASET_Binary_V1.0", and the extracted features saved in “BinaryFeatures” folder. Code to run: “train_binary_classifier.py”. Command to Run: python train_binary_classifier.py --model <model name (vgg16, vgg19, resnet)> --datasetdir --outputdir
	- Step 3: Test Classifiers. Requirements: The dataset "DATASET_Binary_V1.0", and the extracted features in “BinaryFeatures” and models in “BinaryModels” folders under same directory Code to run: “test_binary_classifier.py” Command to Run: python test_binary_classifier.py --model model <model name (vgg16, vgg19, resnet)> --datasetdir --outputdir
		- Example: To test the classifiers the following process needs to be followed: (1) create a directory containing the data - DCCDatasets, (2) copy the directory DATASET_Binary_V1.0 from our dataset repository [DCC](https://osf.io/jdhw8/?view_only=f6ed10613af94c6d8050796a30f1568b), (3) create an empty folder called OutputData in the DCCCatasets folder, (4) copy the folders BinaryFitures and BinaryModels from our data repository [DCC](https://osf.io/jdhw8/?view_only=f6ed10613af94c6d8050796a30f1568b) to the OutputData folder, (5) execute the following command ((note that the vgg19 model is used): 
```
python test_binary_classifier.py --model vgg19 --datasetdir C;/Home/src/DCCDatasets/ --outputdir C:/Home/src/DCCDatasets/OutputData 
```


- **Task 2:** Perform Multiclass classification. The aim here is to employ a six-class neural network classifier to identify the category of the input DL architecture figure. We follow similar steps as described for binary classifier to extract features from the three pretrained models. The relevant code cab be found in the folder: “diagram2graph\FigClassify\MultiClassClassify”
	- Step 1: Extract features. Requirement: The dataset "DATASET_MultiClass_V1.0". Code to run: “compute_features_multiclass.py” Command to Run: python compute_features_multiclass.py --model <model name (vgg16, vgg19, resnet)> --datasetdir --outputdir
	- Step 2: Train classifiers. Requirements: The dataset "DATASET_Binary_V1.0", and the extracted features saved in “MultiClassFeatures” folder. Code to run: “train_ multiclass_classifier.py”. Command to Run: python train_multiclass_classifier.py --model <model name (vgg16, vgg19, resnet)> --datasetdir --outputdir
	- Step 3: Test Classifiers. Requirement: The dataset "DATASET_ MultiClass  _V1.0", and the extracted features in “MultiClassFeatures” and models in “MultiClassModels” folders under same directory. Code to run: “test_  multiclass  _classifier.py”. Command to Run: python test_  multiclass_classifier.py --model model <model name (vgg16, vgg19, resnet)> --datasetdir --outputdir
		- Example: To test the classifiers the following process needs to be followed: (1) create a directory containing the data - DCCDatasets, (2) copy the directory DATASET_Binary_V1.0 from our dataset repository [DCC](https://osf.io/jdhw8/?view_only=f6ed10613af94c6d8050796a30f1568b), (3) create an empty folder called OutputData in the DCCCatasets folder, (4) copy the folders MultiClassFitures and MultiClassModels from our data repository [DCC](https://osf.io/jdhw8/?view_only=f6ed10613af94c6d8050796a30f1568b) to the OutputData folder, (5) execute the following command (note that the vgg19 model is used):
```
python test_multiclass_classifier.py --model vgg19 --datasetdir C;/Home/src/DCCDatasets/ --outputdir C:/Home/src/DCCDatasets/OutputData 
```

- **Task 3:** Detect components of deep learning flow diagram. Folder: “diagram2graph\FigAnalysis\ShapeExtraction”
	- Requirement: The dataset "DATASET_MultiClass_V1.0” Code to run: “detect_shapes.py” Command to Run: python3 detect_shapes.py --inputdir --outputdir

- **Task 4:** Recognize text in deep learning flow diagram. The text in each node/layer is obtained through Optical Character Recognition (OCR) using Tesseract7. Based on our manual observation, we assume that a layer description is present near the detected node (either inside or in nearby region). The relevant code cab be found in the folder: “diagram2graph\FigAnalysis\TextExtraction”
	- Requirement: The dataset "DATASET_MultiClass_V1.0” Code to run: “extract_text.py” Command to Run: python3 extract_text.py --inputdir --outputdir