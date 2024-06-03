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

def cnn1(input_shape=(28, 28, 1), num_classes=10):
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

    cnn1_model = keras.Model(inputs=inputs, outputs=outputs, name="cnn1")
    return cnn1_model

def cnn2(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(.5)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    cnn2_model = keras.Model(inputs=inputs, outputs=outputs, name="cnn2")
    return cnn2_model

def create_model(input_shape=(28, 28, 1)):
    inputs = keras.Input(shape=input_shape)

    # denoiser
    encoder_model = encoder()
    decoder_model = decoder((LATENT_DIM,))
    latents = encoder_model(inputs)
    outputs = decoder_model(latents)

    denoiser_model = keras.Model(inputs=inputs, outputs=outputs, name='autoencoder')

    # Classifier
    cnn1_model = cnn1()
    cnn2_model = cnn2()
    branch_1 = cnn1_model(inputs)
    branch_2 = cnn2_model(inputs)
    x = keras.layers.concatenate([branch_1, branch_2], axis=1)
    cnn_output = keras.layers.Dense(10, activation='softmax')(x)
    classifier = keras.Model(inputs=inputs, outputs=cnn_output, name='classifier')

    # build final model
    input_final = keras.Input(shape=(28, 28, 1))
    denoiser = denoiser_model(input_final)
    classifier_model = classifier(input_final)

    # final model
    auto_encoder = keras.Model(input_final, outputs=[denoiser, classifier_model])
    # auto_encoder = keras.Model(input_final, outputs={'autoencoder': denoiser, 'classifier': classifier_model})

    return auto_encoder

def get_best_model_callback(path=CALLBACK_DIR):
    return keras.callbacks.ModelCheckpoint(filepath=path + '/best_model.tf',
                   save_best_only=True)