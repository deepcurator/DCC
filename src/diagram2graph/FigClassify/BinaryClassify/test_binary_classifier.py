import numpy as np
import h5py
import keras
import os
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import model_from_json
import argparse
from keras.applications import ResNet50 #ip:224*244
from keras.applications import InceptionV3 #ip:299*299
from keras.applications import Xception  #ip:299*299
from keras.applications import VGG16 #ip:224*244
from keras.applications import VGG19 #ip:224*244
from keras.preprocessing.image import ImageDataGenerator, load_img
# Command: python test_diagram_binary_classify.py --model vgg16

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
ap.add_argument("-o", "--outputdir", required=True, help='Path to dir containing "Output"')

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


# saved feature files path
feature_dir_path = os.path.join(args["outputdir"], 'Output/Features')
testFeature_path = os.path.join(feature_dir_path, args["model"]+'_test_features.h5')
testLabel_path = os.path.join(feature_dir_path, args["model"]+'_test_labels.h5')


if not(os.path.isfile(testFeature_path)):
	#Extract the features ## TODO ##
	print ("File %s does not exist" % testFeature_path)

# saved model path
model_dir_path = os.path.join(args["outputdir"], 'Output/Models')
model_json_path = os.path.join(model_dir_path, 'binary_' + args["model"]+'.json')
model_h5_path = os.path.join(model_dir_path, 'binary_' + args["model"]+'.h5')

if not(os.path.isfile(model_json_path)):
	# train the classifier ## TODO ##
	 print ("File %s does not exist" % model_json_path)


if (os.path.isfile(model_json_path) and os.path.isfile(testFeature_path)):
	
	#Load test features
	h5f_testdata  = h5py.File(testFeature_path, 'r')
	h5f_testlabel = h5py.File(testLabel_path, 'r')

	testfeatures_string = h5f_testdata[args["model"]+'_test_features']
	testlabels_string   = h5f_testlabel[args["model"]+'_test_labels']

	test_features = np.array(testfeatures_string)
	test_labels   = np.array(testlabels_string)

	h5f_testdata.close()
	h5f_testlabel.close()

	# verify the shape of features and labels
	print ("[INFO] test features shape: {}".format(test_features.shape))
	print ("[INFO] test labels shape: {}".format(test_labels.shape))
	

	# load json and create model
	json_file = open(model_json_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_h5_path)
	print("Loaded model from disk")
	 
	inputSize = 7*7*512
	inputShape = (7,7,512)
	image_size=(224, 224)
	batch_size = 20
	# if we are using the InceptionV3 or Xception networks, then we need to set the input size
	if args["model"] in ("inception", "xception"):
		image_size =(299, 299)

	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	loss, acc = loaded_model.evaluate(test_features, test_labels)
	print("\nTesting loss: {}, acc: {}\n".format(loss, acc))

	test_dir = 'DATASET_Binary_V1.0/Testing'

	datagen = ImageDataGenerator(rescale=1./255)                   
	test_generator = datagen.flow_from_directory(
	    test_dir,
	    target_size=image_size,
	    batch_size=20,
	    class_mode='categorical',
	    shuffle=False)
	                        
	fnames = test_generator.filenames

	ground_truth = test_generator.classes
	# print ('%s ' % (ground_truth ))

	label2index = test_generator.class_indices
	# print ('%s ' % (label2index))

	# Getting the mapping from class index to class label
	idx2label = dict((v,k) for k,v in label2index.items())
	# print ('%s ' % (idx2label))

	predictions = loaded_model.predict_classes(test_features)
	# print ('%s ' % (predictions))	

	errors = np.where(predictions != ground_truth)[0]
	print("No of errors = {}".format(len(errors)))	
	print('Confusion Matrix')
	print(confusion_matrix(ground_truth, predictions))
	print('Classification Report')
	print(classification_report(ground_truth, predictions, target_names=list(idx2label.values())))
