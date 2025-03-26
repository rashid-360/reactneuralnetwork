import tensorflow as tf
import numpy as np
import json

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1)  # Shape: (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # Shape: (10000, 28, 28, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,        # Rotate images randomly by up to 10 degrees
    width_shift_range=0.1,    # Shift images horizontally by up to 10%
    height_shift_range=0.1,   # Shift images vertically by up to 10%
    zoom_range=0.1            # Zoom images randomly by up to 10%
)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(10, activation='softmax', name='dense_2')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 64
steps_per_epoch = len(x_train) // batch_size  # Ensure steps_per_epoch is an integer

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          steps_per_epoch=steps_per_epoch, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

weights_and_biases = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer_name = layer.name
        weights, biases = layer.get_weights()
        weights_and_biases[layer_name] = {
            'weights': weights.tolist(),
            'biases': biases.tolist()
        }

with open('mnist_weights_biases.json', 'w') as json_file:
    json.dump(weights_and_biases, json_file)

print("Weights and biases have been saved to mnist_weights_biases.json")
