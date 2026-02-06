import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.layers import Input, Flatten, Dense
from keras.models import Model

def get_model(shape=(28, 28, 1)):
    visible = Input(shape=shape)
    flat = Flatten()(visible)
    hidden = Dense(128, activation='relu')(flat)
    hidden = Dense(64, activation='relu')(hidden)
    output = Dense(10, activation='softmax')(hidden)

    model = Model(inputs=visible, outputs=output)

    return model