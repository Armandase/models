import tensorflow as tf
hidden1 = 100
hidden2 = 100
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28, 1)),

            tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden1, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(hidden2, activation='relu'),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.sequence(x)

