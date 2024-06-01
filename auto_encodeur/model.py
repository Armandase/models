import keras
from constants import *

def encoder(input_shape=(28, 28, 1)):
    input_layer = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(input_layer)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    latent_layer = keras.layers.Dense(LATENT_DIM)(x)

    encoder_model = keras.Model(inputs=input_layer, outputs=latent_layer, name="encoder")
    return encoder_model

def decoder(input_shape=(LATENT_DIM,)):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2)(x)
    x = keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(x)
    outputs = keras.layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

    decoder_model = keras.Model(inputs=inputs, outputs=outputs, name="decoder")
    return decoder_model

def cnn(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(.5)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    cnn_model = keras.Model(inputs=inputs, outputs=outputs, name="cnn")
    return cnn_model

def create_model(input_shape=(28, 28, 1)):
    inputs = keras.Input(shape=input_shape)

    encoder_model = encoder()
    decoder_model = decoder((LATENT_DIM,))
    cnn_model = cnn()

    latents = encoder_model(inputs)
    outputs = decoder_model(latents)

    denoiser_model = keras.Model(inputs=inputs, outputs=outputs, name='denoiser')
    denoiser = denoiser_model(inputs)
    conv_model = cnn_model(inputs)

    auto_encoder = keras.Model(inputs=inputs, outputs=[denoiser, conv_model], name='autoencoder')

    return auto_encoder

def get_best_model_callback(path=CALLBACK_DIR):
    return keras.callbacks.ModelCheckpoint(filepath=path + '/best_model.tf',
                   save_best_only=True)