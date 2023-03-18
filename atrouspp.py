# source: https://www.kaggle.com/code/utkarshsaxenadn/water-bodies-segmentation-deeplabv3
# paper: https://arxiv.org/pdf/1606.00915.pdf
import Modules
import numpy as np
# import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, Conv2D, add, TimeDistributed, GlobalAveragePooling2D, \
    Dropout, \
    Concatenate, concatenate, \
    Dense, GlobalMaxPooling2D, MaxPooling2D, Lambda, Add, Layer, BatchNormalization, ReLU, AveragePooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
class ConvBlock(Layer):

    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, use_bias=False,
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])

    def call(self, X):
        return self.net(X)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
        }


def AtrousSpatialPyramidPooling(X):
    B, H, W, C = X.shape

    # Image Pooling
    image_pool = AveragePooling2D(pool_size=(H, W))(X)
    image_pool = ConvBlock(kernel_size=1)(image_pool)
    image_pool = UpSampling2D(size=(H // image_pool.shape[1], W // image_pool.shape[2]),
                              )(image_pool)

    # Atrous Oprtations using dilation
    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1)(X)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6)(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12)(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18)(X)

    # Combine All
    combined = Concatenate()([image_pool, conv_1, conv_6, conv_12, conv_18])
    processed = ConvBlock(kernel_size=1)(combined)

    # Final Output
    return processed