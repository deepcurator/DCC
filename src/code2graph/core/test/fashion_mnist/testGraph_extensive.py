import os
from time import time

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def clearLogFolder():
	folder = "../tmp/log"
	if(os.path.exists(folder)):
		for the_file in os.listdir(folder):
		    file_path = os.path.join(folder, the_file)
		    try:
		        if os.path.isfile(file_path):
		            os.unlink(file_path)
		        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
		    except Exception as e:
		        print(e)
	else:
		try:
			os.mkdir(folder)
		except OSError as os_error:
			print (os_error)


clearLogFolder()

'''Load MNIST Data Set'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''Preprocess Data'''
train_images = train_images / 255.0
test_images = test_images / 255.0

'''Build the Model'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''Compile the Model'''
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''Create a new TensorBoard instance'''
tensorboard = TensorBoard(log_dir="../tmp/log",histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# from tensorflow.keras import backend as K
# session = K.get_session()
# writer = tf.summary.FileWriter("../tmp/log", session.graph)
# import sys
# sys.exit()
'''Train the Model'''
model.fit(train_images, train_labels, epochs=1, callbacks=[tensorboard])

'''Test the Accuracy'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# keras.callbacks.TensorBoard(log_dir='../tmp/log', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
