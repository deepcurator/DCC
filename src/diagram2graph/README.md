This folder contains code for diagram2graph - knowdlege extraction from diagrams of deep neural network architectures.
The datasets refered to can be found at [OSF](https://osf.io/jdhw8/?view_only=f6ed10613af94c6d8050796a30f1568b)

1.	Task 1: Do binary classification. 
Folder: “diagram2graph\FigClassify\BinaryClassify”

a.	Step 1: Extract features
Requirement: The dataset "DATASET_Binary_V1.0" 
Code to run: “compute_features.py”
Command to Run: python compute_features.py --model <model name (vgg16, vgg19, resnet)> --datasetdir <dir where dataset is saved> --outputdir <dir where computed features will be saved>

b.	Step 2: Train classifiers
Requirement: The dataset "DATASET_Binary_V1.0", extracted features saved in “BinaryFeatures” folder
Code to run: “train_binary_classifier.py”
Command to Run: python train_binary_classifier.py --model <model name (vgg16, vgg19, resnet)> --datasetdir <dir where dataset is saved> --outputdir <dir where computed features are saved>

c.	Step 3: Test Classifiers
Requirement: The dataset "DATASET_Binary_V1.0", extracted features in “BinaryFeatures” and models in “BinaryModels” folders under same directory
Code to run: “test_binary_classifier.py”
Command to Run: python test_binary_classifier.py --model model <model name (vgg16, vgg19, resnet)> --datasetdir <dir where dataset is saved> --outputdir <dir where computed features are saved>

2.	Task 2: Do Multiclass classification. 
Folder: “diagram2graph\FigClassify\MultiClassClassify”

a.	Step 1: Extract features
Requirement: The dataset "DATASET_MultiClass_V1.0" 
Code to run: “compute_features_multiclass.py”
Command to Run: python compute_features_multiclass.py --model <model name (vgg16, vgg19, resnet)> --datasetdir <dir where dataset is saved> --outputdir <dir where computed features will be saved>

b.	Step 2: Train classifiers
Requirement: The dataset "DATASET_Binary_V1.0", extracted features saved in “MultiClassFeatures” folder
Code to run: “train_ multiclass_classifier.py”
Command to Run: python train_multiclass_classifier.py --model <model name (vgg16, vgg19, resnet)> --datasetdir <dir where dataset is saved> --outputdir <dir where computed features are saved>

c.	Step 3: Test Classifiers
Requirement: The dataset "DATASET_ MultiClass _V1.0", extracted features in “MultiClassFeatures” and models in “MultiClassModels” folders under same directory
Code to run: “test_ multiclass _classifier.py”
Command to Run: python test_ multiclass_classifier.py --model model <model name (vgg16, vgg19, resnet)> --datasetdir <dir where dataset is saved> --outputdir <dir where computed features are saved>

3.	Task 3: Detect components of deep learning flow diagram. 
Folder: “diagram2graph\FigAnalysis\ShapeExtraction”

Requirement: The dataset "DATASET_MultiClass_V1.0”
Code to run: “detect_shapes.py”
Command to Run: python3 detect_shapes.py --inputdir <dir where dataset is saved> --outputdir <dir where output  will be saved>

4.	Task 4: Recognize text in deep learning flow diagram.
Folder: “diagram2graph\FigAnalysis\TextExtraction”

Requirement: The dataset "DATASET_MultiClass_V1.0”
Code to run: “extract_text.py”
Command to Run: python3 extract_text.py --inputdir <dir where dataset is saved> --outputdir <dir where output  will be saved>
