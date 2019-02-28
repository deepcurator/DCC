import keras
import os
import numpy as np
import argparse
import h5py 
import json
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import ResNet50 #ip:224*244
from keras.applications import InceptionV3 #ip:299*299
from keras.applications import Xception  #ip:299*299
from keras.applications import VGG16 #ip:224*244
from keras.applications import VGG19 #ip:224*244
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# Command: python compute_features_multiclass.py --model vgg16 --datasetdir <dir where dataset is saved> --outputdir <dir where computed featueres will be saved>


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument('-i', '--datasetdir', required=True, help='Path to input dir containing "DATASET_MultiClass_V1.0"')
ap.add_argument("-o", "--outputdir", required=True, help='Path to output dir')
 
args = vars(ap.parse_args())


# Define a dictionary that maps model names to their classes inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, 
	"resnet": ResNet50
}
 
# Ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

# The path for train and test data 

train_dir = os.path.join(args["datasetdir"], 'DATASET_MultiClass_V1.0/Training')
validation_dir = os.path.join(args["datasetdir"], 'DATASET_MultiClass_V1.0/Validation')
test_dir = os.path.join(args["datasetdir"], 'DATASET_MultiClass_V1.0/Testing')

# Output path for saving features
output_dir = os.path.join(args["outputdir"], 'MultiClassFeatures')
if not (os.path.isdir(output_dir)):
    try:  
        os.mkdir(output_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % output_dir)
    else:  
        print ("Successfully created the directory %s " % output_dir)

output_trainingFeature = os.path.join(output_dir, args["model"]+'multiclass_train_features.h5')
output_trainingLabel = os.path.join(output_dir, args["model"]+'multiclass_train_labels.h5')
output_validationFeature = os.path.join(output_dir, args["model"]+'multiclass_validation_features.h5')
output_validationLabel = os.path.join(output_dir, args["model"]+'multiclass_validation_labels.h5')
output_testFeature = os.path.join(output_dir, args["model"]+'multiclass_test_features.h5')
output_testLabel = os.path.join(output_dir, args["model"]+'multiclass_test_labels.h5')

# Initialize the default input image shape (224x224 pixels) and output feature size to (7x7x512) for vgg16/vgg19
nTrain = 1000
nVal = 98
nTest = 98
height_width = 224
inputShape = (height_width, height_width, 3)
trainoutputShape = (nTrain, 7, 7, 512)
outputSize = 7*7*512
validationoutputShape = (nVal, 7, 7, 512) 
testoutputShape = (nTest, 7, 7, 512) 


# If we are using the InceptionV3 or Xception networks, then we need to set the input shape to (299x299) [rather than (224x224)]
if args["model"] in ("resnet"):
	trainoutputShape = (nTrain, 1, 1, 2048)
	validationoutputShape = (nVal, 1, 1, 2048)
	testoutputShape = (nTest, 1, 1, 2048) 
	outputSize = 2048
if args["model"] in ("inception"):
	height_width = 299
	inputShape = (height_width, height_width, 3)
	trainoutputShape = (nTrain, 8, 8, 2048) 
	validationoutputShape = (nVal, 8, 8, 2048)
	testoutputShape = (nTest, 8, 8, 2048) 
	outputSize = 8*8*2048
if args["model"] in ("xception"):
	height_width = 299
	inputShape = (height_width, height_width, 3)
	trainoutputShape = (nTrain, 10, 10, 2048) 
	validationoutputShape = (nVal, 10, 10, 2048)
	testoutputShape = (nTest, 10, 10, 2048) 
	outputSize = 10*10*2048


# Load the network weights pretrained with imagenet

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights='imagenet', include_top=False, input_shape = inputShape)
model.summary()

# Initilize data generator
 
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
train_features = np.zeros(shape=trainoutputShape) 
train_labels = np.zeros(shape=(nTrain,6))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(height_width, height_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = model.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break
        
train_features = np.reshape(train_features, (nTrain, outputSize))
print("Feature Extraction from Training Data Done")
# save features and labels
h5f_traindata = h5py.File(output_trainingFeature, 'w')
h5f_traindata.create_dataset(args["model"]+'multiclass_train_features', data=np.array(train_features))

h5f_trainlabel = h5py.File(output_trainingLabel, 'w')
h5f_trainlabel.create_dataset(args["model"]+'multiclass_train_labels', data=np.array(train_labels))

h5f_traindata.close()
h5f_trainlabel.close()


validation_features = np.zeros(shape=validationoutputShape) 
validation_labels = np.zeros(shape=(nVal,6))

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(height_width, height_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = model.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, outputSize) )
print("Feature Extraction from Validation Data Done")

# save features and labels
h5f_validationdata = h5py.File(output_validationFeature, 'w')
h5f_validationdata.create_dataset(args["model"]+'multiclass_validation_features', data=np.array(validation_features))

h5f_validationlabel = h5py.File(output_validationLabel, 'w')
h5f_validationlabel.create_dataset(args["model"]+'multiclass_validation_labels', data=np.array(validation_labels))

h5f_validationdata.close()
h5f_validationlabel.close()


test_features = np.zeros(shape=testoutputShape) 
test_labels = np.zeros(shape=(nTest,6))

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(height_width, height_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in test_generator:
    features_batch = model.predict(inputs_batch)
    test_features[i * batch_size : (i + 1) * batch_size] = features_batch
    test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTest:
        break

test_features = np.reshape(test_features, (nTest, outputSize) )
print("Feature Extraction from Test Data Done")

# save features and labels
h5f_testdata = h5py.File(output_testFeature, 'w')
h5f_testdata.create_dataset(args["model"]+'multiclass_test_features', data=np.array(test_features))

h5f_testlabel = h5py.File(output_testLabel, 'w')
h5f_testlabel.create_dataset(args["model"]+'multiclass_test_labels', data=np.array(test_labels))

h5f_testdata.close()
h5f_testlabel.close()

# verify the shape of features and labels
print ("[INFO] train features shape: {}".format(train_features.shape))
print ("[INFO] train labels shape: {}".format(train_labels.shape))
print ("[INFO] validation features shape: {}".format(validation_features.shape))
print ("[INFO] validation labels shape: {}".format(validation_labels.shape))
print ("[INFO] test features shape: {}".format(test_features.shape))
print ("[INFO] test labels shape: {}".format(test_labels.shape))



