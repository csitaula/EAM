from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D, Add, \
    Concatenate,Multiply
from tensorflow.keras import backend as K
from cbam_attention import cbam_block_improved


def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):
    ''' Create a spatial squeeze-excite block
    Args:
        input: input tensor
    Returns: a keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x


# Proposed algorithm for mid-level feature extraction
def EACRM(input_tensor):
    lev_1 = Conv2D(256, (1, 1))(input_tensor)  # 1x1 convolution and reduce channel
    print(lev_1.shape)
    lev_2 = Conv2D(256, (3, 3))(input_tensor)  # 3x3 convolution and reduce channel
    # lev_2= GlobalAveragePooling2D()(lev_2)
    # print(lev_2.shape)
    lev_3 = cbam_block_improved(input_tensor)  # CBAM modified block
    print(lev_3.shape)
    # res1 = multiply([lev_1, lev_3])
    res1 = Concatenate()(
        [lev_1, lev_3])
    return res1


def conv1by1(input_tensor):
    tensor = Conv2D(256, (1, 1))(input_tensor)
    return tensor
