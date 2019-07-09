import tensorflow as tf
import os

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

x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.constant([[37.0, -23.0], [1.0, 4.0]])
c = x+w
y = tf.matmul(x, w)
z = tf.matmul(y, c)
init= tf.global_variables_initializer()



with tf.Session() as sess:
	sess.run(init)
	print(sess.run(z))
	writer = tf.summary.FileWriter("../tmp/log/", sess.graph)
	writer.close()
