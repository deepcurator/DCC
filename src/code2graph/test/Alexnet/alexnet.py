import tensorflow as tf
from tensorflow import keras


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    model = keras.Sequential([
    keras.layers.Conv2D(64, 3, 11, 11, border_mode='full'),
    keras.layers.BatchNormalization((64,226,226)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(poolsize=(3, 3)),

    keras.layers.Conv2D(128, 64, 7, 7, border_mode='full'),
    keras.layers.BatchNormalization((128,115,115)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(poolsize=(3, 3)),
        keras.layers.Conv2D(192, 128, 3, 3, border_mode='full'),
    keras.layers.BatchNormalization((128,112,112)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(poolsize=(3, 3)),

    keras.layers.Conv2D(256, 192, 3, 3, border_mode='full'),
    keras.layers.BatchNormalization((128,108,108)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(poolsize=(3, 3)),

    keras.layers.Flatten(),
    keras.layers.Dense(12*12*256, 4096, init='normal'),
    keras.layers.BatchNormalization(4096),
    keras.layers.Activation('relu'),
    keras.layers.Dense(4096, 4096, init='normal'),
    keras.layers.BatchNormalization(4096),
    keras.layers.Activation('relu'),
    keras.layers.Dense(4096, 1000, init='normal'),
    keras.layers.BatchNormalization(1000),
    keras.layers.Activation('softmax')])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy')

    model.fit(x_train, y_train, epochs=1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

if __name__ == "__main__" :
    main()
