import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, TimeDistributed, LSTM
from tensorflow.keras.applications import InceptionV3

def create_sibi_model(num_classes):
    """
    Creates a 3D-CNN model with InceptionV3 for SIBI recognition.
    """
    # Input layer
    input_shape = (10, 100, 100, 3)  # 10 frames, 100x100 image, 3 channels
    input_layer = Input(shape=input_shape)

    # TimeDistributed InceptionV3
    base_inception = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    # Make InceptionV3 layers non-trainable
    for layer in base_inception.layers:
        layer.trainable = False
    time_distributed_inception = TimeDistributed(base_inception)(input_layer)

    # LSTM layer
    lstm_layer = LSTM(256)(time_distributed_inception)

    # Dense layers
    dense_1 = Dense(128, activation='relu')(lstm_layer)
    output_layer = Dense(num_classes, activation='softmax')(dense_1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

if __name__ == '__main__':
    num_classes = 25  # A to Y
    model = create_sibi_model(num_classes)
    model.summary()
