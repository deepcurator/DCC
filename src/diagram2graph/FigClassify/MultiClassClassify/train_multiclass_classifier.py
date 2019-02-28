import numpy as np
import h5py
import keras
import argparse
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50 #ip:224*244
from keras.applications import InceptionV3 #ip:299*299
from keras.applications import Xception  #ip:299*299
from keras.applications import VGG16 #ip:224*244
from keras.applications import VGG19 #ip:224*244
from keras.preprocessing import image
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint



# Command: python train_multiclass_classifier.py --model vgg16 --datasetdir <dir where dataset is saved> --outputdir <dir where computed features are saved>

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
ap.add_argument('-i', '--datasetdir', required=True, help='Path to dir containing "DATASET_MultiClass_V1.0"')
ap.add_argument("-o", "--outputdir", required=True, help='Path to dir containing "MultiClassFeatures"')
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


# Feature file path
feature_dir_path = os.path.join(args["outputdir"], 'MultiClassFeatures')
trainingFeature_path = os.path.join(feature_dir_path, args["model"]+'multiclass_train_features.h5')
trainingLabel_path = os.path.join(feature_dir_path, args["model"]+'multiclass_train_labels.h5')
validationFeature_path = os.path.join(feature_dir_path, args["model"]+'multiclass_validation_features.h5')
validationLabel_path = os.path.join(feature_dir_path, args["model"]+'multiclass_validation_labels.h5')

validation_dir = os.path.join(args["datasetdir"], 'DATASET_MultiClass_V1.0/Validation')

# Path for saving trained model
model_dir_path = os.path.join(args["outputdir"], 'MultiClassModels')
if not (os.path.isdir(model_dir_path)):
    try:  
        os.mkdir(model_dir_path)
    except OSError:  
        print ("Creation of the directory %s failed" % model_dir_path)
    else:  
        print ("Successfully created the directory %s " % model_dir_path)

model_json_path = os.path.join(model_dir_path, 'multiclass_' + args["model"]+'.json')
model_h5_path = os.path.join(model_dir_path, 'multiclass_' + args["model"]+'.h5')


###############################################################################################################
nTrain = 1000
nVal = 98
batch_size = 20

#Load training and validation features

h5f_traindata  = h5py.File(trainingFeature_path, 'r')
h5f_trainlabel = h5py.File(trainingLabel_path, 'r')

trainfeatures_string = h5f_traindata[args["model"]+'multiclass_train_features']
trainlabels_string   = h5f_trainlabel[args["model"]+'multiclass_train_labels']

train_features = np.array(trainfeatures_string)
train_labels   = np.array(trainlabels_string)

h5f_traindata.close()
h5f_trainlabel.close()

h5f_validationdata  = h5py.File(validationFeature_path, 'r')
h5f_validationlabel = h5py.File(validationLabel_path, 'r')

validationfeatures_string = h5f_validationdata[args["model"]+'multiclass_validation_features']
validationlabels_string   = h5f_validationlabel[args["model"]+'multiclass_validation_labels']

validation_features = np.array(validationfeatures_string)
validation_labels   = np.array(validationlabels_string)

h5f_validationdata.close()
h5f_validationlabel.close()

# verify the shape of features and labels
print ("[INFO] train features shape: {}".format(train_features.shape))
print ("[INFO] train labels shape: {}".format(train_labels.shape))
print ("[INFO] validation features shape: {}".format(validation_features.shape))
print ("[INFO] validation labels shape: {}".format(validation_labels.shape))

###############################################################################################################

inputSize = 7*7*512
inputShape = (7, 7, 512)
image_size=(224, 224)
learning_rate = 2e-6
epoch_no = 1000
# if we are using the InceptionV3 or Xception networks, then we need to set the input size
if args["model"] in ("resnet"):
	inputSize = 2048
	inputShape = (1, 1, 2048)
	
if args["model"] in ("inception"):
	inputSize = 8*8*2048
	inputShape = (8, 8, 2048)
	image_size =(299, 299)
if args["model"] in ("xception"):
	inputSize = 10*10*2048
	inputShape = (10, 10, 2048)
	image_size =(299, 299)


model = models.Sequential()
if args["model"] in ("resnet"):
	model.add(layers.Dense(512, activation='relu', input_dim=inputSize))
	epoch_no = 200
if args["model"] in ("inception", "xception"):
	model.add(layers.Dense(2048, activation='relu', input_dim=inputSize))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.25))
	
	#model.add(layers.Dense(2048, activation='relu', input_dim=inputSize))
	#model.add(layers.Dropout(0.25))
	#model.add(layers.Dense(1024, activation='relu'))
	#model.add(layers.Dropout(0.5))
	#model.add(layers.Dense(256, activation='relu'))
else:
	model.add(layers.Dense(1024, activation='relu', input_dim=inputSize))
	#model.add(layers.Dropout(0.5))
	#model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['acc'])

callbacks = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')        
# autosave best Model
best_model = ModelCheckpoint(model_h5_path, monitor='val_loss', verbose = 1, save_best_only = True)


history = model.fit(train_features,
                    train_labels,
                    epochs=epoch_no,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels),
		    callbacks = [callbacks,best_model])
 
# serialize model to JSON
model_json = model.to_json()
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
#model.save_weights(model_h5_path)
#print("Saved model to disk")

# summarize history for accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc']); plt.plot(history.history['val_acc']);
plt.title('model accuracy'); plt.ylabel('accuracy');
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');

# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss']); plt.plot(history.history['val_loss']);
plt.title('model loss'); plt.ylabel('loss');
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');
plt.show()


