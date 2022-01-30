from typing import Tuple


from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv3D, TimeDistributed, Input, Dense, Flatten, ConvLSTM2D, AveragePooling3D, Reshape

def DeepMotionCLF_MD(shape: Tuple = (500, 13, 13, 2),
                     classes=1,
                     use_bias=True,
                     mode: str = "training"):
    """ Simple LSTM model. In this version, no average pooling is used and shape is reduced by convolutions.

    # Argument:
        - shape: The input shape.
        - classes: The number of output classes.
        - use_bias: If the network should use bias for all the layers.
    """
    input = Input(shape=shape, name="data_input")

    first_convolution = Conv2D(32, 3, 2, use_bias=use_bias, activation="relu")
    second_convolution = Conv2D(32, 3, 2, use_bias=use_bias, activation="relu")

    x = TimeDistributed(first_convolution)(input)
    x = TimeDistributed(second_convolution)(x)

    x = ConvLSTM2D(32, 2, use_bias=use_bias)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu", name="latent_output")(x)
    x = Dense(classes, activation="relu", use_bias=use_bias, name="predictions")(x)

    return Model(input, x)

def DeepMotionCLS_MD(shape: Tuple = (20, 25, 13, 13, 2),
                        classes: int = 1,
                        use_bias: bool = True,
                        mode="training") -> object:
    """ ConvLSTM model based on the the usage of grouped frames as input. Designed for the Moving Digit shaped inputs (200, 200).

    # Argument:
        - shape: The input shape.
        - classes: The number of output classes.
        - use_bias: If the network should use biases for all the layers.
        - mode: If the model should be set for training or inference (not used here).
    """
    input = Input(shape=shape, name="data_input")

    # Layers used for feature extraction
    first_convolution = Conv3D(64, 3, 2, use_bias=use_bias, activation="relu")
    second_convolution = Conv3D(64, 3, 2, use_bias=use_bias, activation="relu")
    temporal_pooling = AveragePooling3D((5, 1, 1))

    # Apply the layers
    x = TimeDistributed(first_convolution)(input)
    x = TimeDistributed(second_convolution)(x)
    x = TimeDistributed(temporal_pooling)(x)

    # Reshape for ConvLSTM processing
    x = Reshape((20, 2, 2, 64))(x)

    # Temporal processing and prediction
    x = ConvLSTM2D(64, 2, use_bias=use_bias)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu", name="latent_output")(x)
    x = Dense(classes,
              activation="relu",
              use_bias=use_bias,
              name="predictions")(x)

    return Model(input, x)

def DeepMotion3D_MD(shape: Tuple = (13, 13, 500, 2),
                   classes=1,
                   use_bias=True,
                   mode="training"):
    """  Model based on the usage of 3D convolutions. Designed for the Moving Digit shaped inputs (200, 200).

    # Argument:
        - shape: The input shape.
        - classes: The number of output classes.
        - use_bias: If the network should use biases for all the layers.
        - mode: If the model should be set for training or inference (not used here).
    """

    input = Input(shape=shape, name="data_input")

    x = Conv3D(64, 3, 2, use_bias=use_bias, activation="relu")(input)
    x = Conv3D(64, 3, 2, use_bias=use_bias, activation="relu")(x)

    x = AveragePooling3D((2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu", name="latent_output")(x)
    x = Dense(classes,
              activation="relu",
              use_bias=use_bias,
              name="predictions")(x)

    return Model(input, x)