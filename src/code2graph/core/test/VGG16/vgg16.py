import tensorflow as tf
from tensorflow import keras
import os

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = keras.Sequential([
        # keras.layers.Conv2D(64, 3, 3, input_shape=(3, 224, 224), activation='relu'),
        keras.layers.ZeroPadding2D((1, 1), input_shape=(3, 224, 224)),
        keras.layers.Conv2D(64, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(64, 3, 3, activation='relu'),
        keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(128, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(128, 3, 3, activation='relu'),
        keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(256, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(256, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(256, 3, 3, activation='relu'),
        keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(512, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(512, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(512, 3, 3, activation='relu'),
        keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(512, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(512, 3, 3, activation='relu'),
        keras.layers.ZeroPadding2D((1, 1)),
        keras.layers.Conv2D(512, 3, 3, activation='relu'),
        keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1000, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy')

    model.fit(x_train, y_train, epochs=1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

if __name__ == "__main__" :
    main()
