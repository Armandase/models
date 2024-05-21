import keras


def create_model(vect_size=10000):
    return keras.Sequential(
        [
            keras.layers.Input(shape=vect_size),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )

def get_best_model_callback(path='/home/armand/projets/neural_networks/imdb1/models'):
    return keras.callbacks.ModelCheckpoint(filepath=path + '/best_model.tf',
                 monitor='val_accuracy',
                 mode='max',
                 save_best_only=True)