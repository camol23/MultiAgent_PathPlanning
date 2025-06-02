#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    23-Apr-2025 13:38:42

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    obsInLyr = keras.Input(shape=(8,))
    actInLyr = keras.Input(shape=(2,))
    cat = layers.Concatenate(axis=1)([obsInLyr, actInLyr])
    fc_1 = layers.Dense(256, name="fc_1_")(cat)
    relu_1 = layers.ReLU()(fc_1)
    fc_2 = layers.Dense(256, name="fc_2_")(relu_1)
    relu_2 = layers.ReLU()(fc_2)
    fc_3 = layers.Dense(512, name="fc_3_")(relu_2)
    relu_3 = layers.ReLU()(fc_3)
    fc_4 = layers.Dense(512, name="fc_4_")(relu_3)
    relu_4 = layers.ReLU()(fc_4)
    fc_5 = layers.Dense(512, name="fc_5_")(relu_4)
    relu_5 = layers.ReLU()(fc_5)
    fc_6 = layers.Dense(256, name="fc_6_")(relu_5)
    relu_6 = layers.ReLU()(fc_6)
    fc_7 = layers.Dense(256, name="fc_7_")(relu_6)
    relu_7 = layers.ReLU()(fc_7)
    fc4 = layers.Dense(1, name="fc4_")(relu_7)

    model = keras.Model(inputs=[obsInLyr, actInLyr], outputs=[fc4])
    return model
