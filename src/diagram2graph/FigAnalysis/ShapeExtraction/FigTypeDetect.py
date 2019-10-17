import numpy as np
import h5py
import keras
import os
from keras.models import model_from_json
import argparse
from keras.applications import ResNet50 #ip:224*244
from keras.applications import InceptionV3 #ip:299*299
from keras.applications import Xception  #ip:299*299
from keras.applications import VGG16 #ip:224*244
from keras.applications import VGG19 #ip:224*244
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.inception_v3 import preprocess_input
import cv2
import json


 

class FigTypeDetect:
    
	def __init__(self):
    	# define a dictionary that maps model names to their classes inside Keras
		self.MODELS = { "vgg16": VGG16, "vgg19": VGG19, "inception": InceptionV3, "xception": Xception, "resnet": ResNet50 }

		self.inputShape = (224, 224)
		# saved model path
		self.model_path = 'Models'

		self.Binaryclass_loaded_model = []
		self.Multiclass_loaded_model = []

		
	def loadFigClassModels(self, modeltype):
		# esnure a valid model name was supplied via command line argument
		if modeltype not in self.MODELS.keys():
			raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

		Binaryclass_model_json_path = os.path.join(self.model_path, 'binary_' + modeltype + '.json')
		Binaryclass_model_h5_path = os.path.join(self.model_path, 'binary_' + modeltype + '.h5')
		Multiclass_model_json_path = os.path.join(self.model_path, 'multiclass_' + modeltype + '.json')
		Multiclass_model_h5_path = os.path.join(self.model_path, 'multiclass_' + modeltype + '.h5')
		# load binary classifier json and create model
		Binaryclass_json_file = open(Binaryclass_model_json_path, 'r')
		Binaryclass_loaded_model_json = Binaryclass_json_file.read()
		Binaryclass_json_file.close()
		self.Binaryclass_loaded_model = model_from_json(Binaryclass_loaded_model_json)

		# load weights into new model
		self.Binaryclass_loaded_model.load_weights(Binaryclass_model_h5_path)
		print("Loaded binary classifier model from disk")
		 
		# load multiclass classifier json and create model
		Multiclass_json_file = open(Multiclass_model_json_path, 'r')
		Multiclass_loaded_model_json = Multiclass_json_file.read()
		Multiclass_json_file.close()
		self.Multiclass_loaded_model = model_from_json(Multiclass_loaded_model_json)

		# load weights into new model
		self.Multiclass_loaded_model.load_weights(Multiclass_model_h5_path)
		print("Loaded multiclass classifier model from disk")
		 


	def detectFigType(self, img):

		
		#print("[INFO] loading and pre-processing image...")
		
		# resize image
		image = cv2.resize(img, self.inputShape, interpolation = cv2.INTER_AREA)
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)
		feature_model = VGG16(weights='imagenet', include_top=False)
		vgg16_feature = feature_model.predict(image)
		test_vgg16_features = np.reshape(vgg16_feature, (1, 7*7*512) )

		
		# Predict the probability across all binary classes
		yhat_binary = self.Binaryclass_loaded_model.predict(test_vgg16_features)
		binary_cls = yhat_binary.argmax(axis=-1)
		
		# Predict the probability across all 6 classes
		yhat_multi = self.Multiclass_loaded_model.predict(test_vgg16_features)
		multi_cls = yhat_multi.argmax(axis=-1)
		
		return binary_cls, multi_cls
			


