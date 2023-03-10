"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""
import Modules
import numpy as np
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, Conv2D, add, TimeDistributed, GlobalAveragePooling2D, Dropout, \
    Concatenate, concatenate, \
    Dense, GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Lambda, Add, LSTM, GRU, Reshape, DepthwiseConv2D, \
    Concatenate, \
    Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
#rom tensorflow.keras.applications import MobileNetV2, VGG16, EfficientNetB0, ResNet50
from tensorflow.keras.applications import ResNet50
import os, math, time
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.utils.vis_utils import plot_model
import tensorflow as tf
# import preprocess_crop

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)


# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# def relu6(x):
#     """Relu 6
#     """
#     return K.relu(x, max_value=6.0)
#
#
# def _conv_block(inputs, filters, kernel, strides):
#     """Convolution Block
#     This function defines a 2D convolution operation with BN and relu6.
#
#     # Arguments
#         inputs: Tensor, input tensor of conv layer.
#         filters: Integer, the dimensionality of the output space.
#         kernel: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#         strides: An integer or tuple/list of 2 integers,
#             specifying the strides of the convolution along the width and height.
#             Can be a single integer to specify the same value for
#             all spatial dimensions.
#
#     # Returns
#         Output tensor.
#     """
#
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#
#     x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
#     x = BatchNormalization(axis=channel_axis)(x)
#     return Activation(relu6)(x)
#
#
# def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
#     """Bottleneck
#     This function defines a basic bottleneck structure.
#
#     # Arguments
#         inputs: Tensor, input tensor of conv layer.
#         filters: Integer, the dimensionality of the output space.
#         kernel: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#         t: Integer, expansion factor.
#             t is always applied to the input size.
#         s: An integer or tuple/list of 2 integers,specifying the strides
#             of the convolution along the width and height.Can be a single
#             integer to specify the same value for all spatial dimensions.
#         alpha: Integer, width multiplier.
#         r: Boolean, Whether to use the residuals.
#
#     # Returns
#         Output tensor.
#     """
#
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     # Depth
#     tchannel = K.int_shape(inputs)[channel_axis] * t
#     # Width
#     cchannel = int(filters * alpha)
#
#     x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
#
#     x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#     x = Activation(relu6)(x)
#
#     x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#
#     if r:
#         x = Add()([x, inputs])
#
#     return x
#
#
# def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
#     """Inverted Residual Block
#     This function defines a sequence of 1 or more identical layers.
#
#     # Arguments
#         inputs: Tensor, input tensor of conv layer.
#         filters: Integer, the dimensionality of the output space.
#         kernel: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#         t: Integer, expansion factor.
#             t is always applied to the input size.
#         alpha: Integer, width multiplier.
#         s: An integer or tuple/list of 2 integers,specifying the strides
#             of the convolution along the width and height.Can be a single
#             integer to specify the same value for all spatial dimensions.
#         n: Integer, layer repeat times.
#
#     # Returns
#         Output tensor.
#     """
#
#     x = _bottleneck(inputs, filters, kernel, t, alpha, strides)
#
#     for i in range(1, n):
#         x = _bottleneck(x, filters, kernel, t, alpha, 1, True)
#
#     return x
#
#
# def MobileNetv2_(input_shape, k, alpha=1.0):
#     """MobileNetv2
#     This function defines a MobileNetv2 architectures.
#
#     # Arguments
#         input_shape: An integer or tuple/list of 3 integers, shape
#             of input tensor.
#         k: Integer, number of classes.
#         alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
#
#     # Returns
#         MobileNetv2 model.
#     """
#     inputs = Input(shape=input_shape)
#
#     first_filters = _make_divisible(32 * alpha, 8)
#     x = _conv_block(inputs, first_filters, (2, 2), strides=(2, 2))
#
#     x = _inverted_residual_block(x, 16, (2, 2), t=1, alpha=alpha, strides=1, n=1)
#     # x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
#     x = _inverted_residual_block(x, 32, (2, 2), t=6, alpha=alpha, strides=2, n=3)
#     x = _inverted_residual_block(x, 64, (2, 2), t=6, alpha=alpha, strides=2, n=4)
#     # x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
#     # x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
#     # x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)
#
#     if alpha > 1.0:
#         last_filters = _make_divisible(1280 * alpha, 8)
#     else:
#         last_filters = 1280
#
#     x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
#     x = GlobalAveragePooling2D()(x)
#     x = Reshape((1, 1, last_filters))(x)
#     x = Dropout(0.3, name='Dropout')(x)
#     x = Conv2D(k, (1, 1), padding='same')(x)
#
#     x = Activation('softmax', name='softmax')(x)
#     output = Reshape((k,))(x)
#
#     model = Model(inputs, output)
#     # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
#
#     return model


# learning decay rate schedule

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.4  # in 0.5 it provided an accuracy of 80%+
    epochs_drop = 4.0  # 5.0 gives an optimal epochs_drop
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


import tensorflow as tf


# def outer_product(x):
#     # Einstein Notation  [batch,1,1,depth] x [batch,1,1,depth] -> [batch,depth,depth]
#     phi_I = tf.einsum('ijkm,ijkn->imn', x[0], x[1])
#
#     # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
#     phi_I = tf.reshape(phi_I, [-1, x[0].shape[3] * x[1].shape[3]])
#
#     # Divide by feature map size [sizexsize]
#     size1 = int(x[1].shape[1])
#     size2 = int(x[1].shape[2])
#     phi_I = tf.divide(phi_I, size1 * size2)
#
#     # Take signed square root of phi_I
#     y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
#
#     # Apply l2 normalization
#     z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
#     return z_l2


# class CancelOut(tensorflow.keras.layers.Layer):
#     '''
#     CancelOut Layer
#     '''
#
#     def __init__(self, activation='sigmoid', cancelout_loss=True, lambda_1=0.002, lambda_2=0.001):
#         super(CancelOut, self).__init__()
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.cancelout_loss = cancelout_loss
#
#         if activation == 'sigmoid': self.activation = tf.sigmoid
#         if activation == 'softmax': self.activation = tf.nn.softmax
#
#     def build(self, input_shape):
#         self.w = self.add_weight(
#             shape=(input_shape[-1],),
#             initializer=tf.keras.initializers.Constant(1),
#             trainable=True)
#
#     def call(self, inputs):
#         if self.cancelout_loss:
#             self.add_loss(self.lambda_1 * tf.norm(self.w, ord=1) + self.lambda_2 * tf.norm(self.w, ord=2))
#         return tf.math.multiply(inputs, self.activation(self.w))
#
#     def get_config(self):
#         return {"activation": self.activation}

# def custom_mobilenetv2(model, classes):
#     # invert1 = model.get_layer('block_11_add').output
#     # invert1=Conv2D(96,(2,2),activation='relu',strides=2)(invert1)
#     # invert2 = model.get_layer('block_12_add').output
#     # invert2= Conv2D(96,(2,2),activation='relu',strides=2)(invert2)
#     invert3 = model.get_layer('block_14_add').output
#     invert4 = model.get_layer('block_15_add').output
#     invert5 = model.get_layer('out_relu').output
#     #invert5 = Conv2D(160, (1, 1), activation='relu')(invert5)
#     # reduce the dimension of tensors having 1280 channel
#     # invert5 = Conv2D(160, (1, 1), activation='relu')(invert5)
#
#     # SE block
#     # invert1_ = Modules.spatial_attention(invert1)
#     # invert2_ = Modules.spatial_attention(invert2)
#     # invert3_ = Modules.spatial_squeeze_excite_block(invert3)
#     # invert4_ = Modules.spatial_squeeze_excite_block(invert4)
#     # invert5_ = Modules.spatial_squeeze_excite_block(invert5)
#     #
#     # # pool
#     # invert1__ = GlobalAveragePooling2D()(invert1_)
#     # invert2__ = GlobalAveragePooling2D()(invert2_)
#     # invert3__ = GlobalAveragePooling2D()(invert3_)
#     # invert4__ = GlobalAveragePooling2D()(invert4_)
#     # invert5__ = GlobalAveragePooling2D()(invert5_)
#     #
#     # # pool the layer
#     # invert1_ = GlobalAveragePooling2D()(invert1)
#     # invert2_ = GlobalAveragePooling2D()(invert2)
#     # invert3_ = GlobalAveragePooling2D()(invert3)
#     # invert4_ = GlobalAveragePooling2D()(invert4)
#     # invert5_ = GlobalAveragePooling2D()(invert5)
#
#     # combine all of them
#     # comb = concatenate([
#     #     # invert1_,
#     #     # invert2_,
#     #     invert3_,
#     #     invert4_,
#     #     invert5_,
#     #     # invert1__,
#     #     # invert2__,
#     #     invert3__,
#     #     invert4__,
#     #     invert5__
#     # ])
#
#     # combine in tensor form
#     comb_ = Concatenate()([invert3, invert4, invert5])
#     print(comb_.shape)
#     comb = Conv2D(1600, (2, 2), activation='relu')(comb_)
#     # comb=MaxPooling2D(pool_size=(2,2),strides=2)(comb)
#     comb = GlobalAveragePooling2D()(comb)
#     # lstm_layer = LSTM(1600, input_shape=(49, 1600), return_sequences=True)(comb)
#     # dense=Dense(1600,activation='relu')(lstm_layer)
#     # reshape = Reshape((49, 1600))(comb_)
#     # lstm_layer = LSTM(1600, input_shape=(49, 1600), return_sequences=True)(reshape)
#     # reshape = Reshape((49, 1280))(invert5)
#     # lstm_layer = LSTM(1280, input_shape=(49, 1280), return_sequences=True)(reshape)
#     # lstm_layer1 = Flatten()(lstm_layer)
#
#     # dense=add([comb,lstm_layer1])
#     # dense=comb
#
#     # comb_=Concatenate()([invert3_,invert4_,invert5_])
#     # comb_=Conv2D(1600,(2,2),activation='relu')(comb_)
#     # comb_=GlobalAveragePooling2D()(comb_)
#
#     # concatenate
#     # comb__=concatenate([comb,comb_])
#
#     # comb=CancelOut(activation='sigmoid')(comb)
#     # # comb=BatchNormalization()(comb) # added to normalize
#     dense = Dense(1024, activation='relu')(comb)  # reduced the 1024->768 (93.19% accuracy), 1024->1024 (93.50%)
#     dense = Dense(1024, activation='relu')(dense)
#     # dense = Lambda(outer_product, name='outer_product')([lstm_layer, invert5])
#     # softmax
#     output = Dense(classes, activation='softmax')(dense)
#     model = Model(inputs=model.input, outputs=output)
#     return model


def custom_resnet50(model, classes):
    invert11 = model.get_layer('conv2_block3_out').output
    invert1 = model.get_layer('conv3_block4_out').output
    invert2 = model.get_layer('conv4_block6_out').output
    invert3 = model.get_layer('conv5_block3_out').output

    # invert11 = model.get_layer('block1_pool').output
    # invert1 = model.get_layer('block2_pool').output
    # invert2 = model.get_layer('block4_pool').output
    # invert3 = model.get_layer('block5_pool').output

    # invert1 = Conv2D(256, (1, 1), activation='relu')(invert1)  # reduce the dimension to 256
    # invert2 = Conv2D(256, (1, 1), activation='relu')(invert2)  # reduce the dimension to 256
    # invert3 = Conv2D(256, (1, 1), activation='relu')(invert3)  # reduce the dimensio to 256
    # invert4 = model.get_layer('block_15_add').output
    # invert5 = model.get_layer('out_relu').output
    # reduce the dimension of tensors having 1280 channel
    # invert5 = Conv2D(160, (1, 1), activation='relu')(invert5)

    # SE block

    invert11_ = Modules.spatial_squeeze_excite_block(invert11)
    invert1_ = Modules.spatial_squeeze_excite_block(invert1)
    invert2_ = Modules.spatial_squeeze_excite_block(invert2)
    invert3_ = Modules.spatial_squeeze_excite_block(invert3)
    # invert4_ = Modules.spatial_squeeze_excite_block(invert4)
    # invert5_ = Modules.spatial_squeeze_excite_block(invert5)

    # pool
    invert11__ = GlobalAveragePooling2D()(invert11_)
    invert1__ = GlobalAveragePooling2D()(invert1_)
    invert2__ = GlobalAveragePooling2D()(invert2_)
    invert3__ = GlobalAveragePooling2D()(invert3_)

    # pool the layer
    invert11_ = GlobalAveragePooling2D()(invert11)
    invert1_ = GlobalAveragePooling2D()(invert1)
    invert2_ = GlobalAveragePooling2D()(invert2)
    invert3_ = GlobalAveragePooling2D()(invert3)

    # combine all of them
    comb = concatenate([
        # invert11_,
        # invert1_,
        # invert2_,
        # invert3_,
        # invert4_,
        # invert5_,
        invert11_,
        invert1_,
        invert2_,
        invert3_,
        # invert4__,
        # invert5__
    ])
    # comb = BatchNormalization()(comb)  # added to normalize
    dense = Dense(728, activation='relu')(comb)  # reduced the 1024->768
    # softmax
    output = Dense(classes, activation='softmax')(dense)
    model = Model(inputs=model.input, outputs=output)
    return model


# def gru_lstm_mobilenetv2(model, classes):
#     invert3 = model.get_layer('block_14_add').output
#     invert4 = model.get_layer('block_15_add').output
#     invert5 = model.get_layer('out_relu').output
#     reshape3 = Reshape((49, 160))(invert3)
#     reshape4 = Reshape((49, 160))(invert4)
#     reshape5 = Reshape((49, 1280))(invert5)
#     # combine in tensor form
#     # comb_ = Concatenate()([invert3, invert4, invert5])
#     # reshape = Reshape((49, 1280))(invert5)
#     lstm_layer = LSTM(1280, input_shape=(49, 1280), return_sequences=True)(reshape3)
#     print(lstm_layer.shape)
#     # r = Reshape((7, 7, 1280))(lstm_layer)
#     lstm_layer3 = Flatten()(lstm_layer)
#     # lstm_layer2 = GlobalMaxPooling2D()(r)
#
#     gru_layer = LSTM(160, input_shape=(49, 160), return_sequences=True)(reshape4)
#     # r = Reshape((7, 7, 160))(gru_layer)
#     lstm_layer4 = Flatten()(gru_layer)
#     # gru_layer2 = GlobalMaxPooling2D()(r)
#
#     gru_layer = LSTM(160, input_shape=(49, 160), return_sequences=True)(reshape5)
#     # r = Reshape((7, 7, 160))(gru_layer)
#     lstm_layer5 = Flatten()(gru_layer)
#     # gru_layer2 = GlobalMaxPooling2D()(r)
#
#     # combine
#     dense = concatenate([lstm_layer3,
#                          lstm_layer4,
#                          lstm_layer5
#                          ])
#     # dense=comb
#     # dense = BatchNormalization()(dense)  # added to normalize
#     dense = Dropout(0.2)(dense)
#     dense = Dense(1024, activation='relu')(dense)  # reduced the 1024->768 (93.19% accuracy), 1024->1024 (93.50%)
#     dense = Dense(1024, activation='relu')(dense)
#     # dense = Lambda(outer_product, name='outer_product')([lstm_layer, invert5])
#     # softmax
#     output = Dense(classes, activation='softmax')(dense)
#     model = Model(inputs=model.input, outputs=output)
#     return model
#
#
# def fine_tune(model, classes):
#     # dense=model.output
#     # dense=GlobalAveragePooling2D()(dense)
#     # dense=BatchNormalization()(dense)
#     # dense=Dense(512,activation='relu')(dense)
#     # output=Dense(classes,activation='softmax')(dense)
#     # model=Model(inputs=model.input,outputs=output)
#     # return model
#     output = model.output
#     # with attention
#     # output = Conv2D(160, (1, 1), activation='relu')(output)  # reduce dimension
#     output1 = Modules.spatial_squeeze_excite_block(output)
#     reshape = Reshape((49, 1280))(output1)
#     lstm_layer = GRU(1280, input_shape=(49, 1280), return_sequences=True)(reshape)
#     lstm_layer1 = Reshape((7, 7, 1280))(lstm_layer)
#     # flatten1 = Flatten()(lstm_layer)
#
#     # without attention
#     reshape = Reshape((49, 1280))(output)
#     lstm_layer = GRU(1280, input_shape=(49, 1280), return_sequences=True)(reshape)
#     lstm_layer2 = Reshape((7, 7, 1280))(lstm_layer)
#     # flatten2 = Flatten()(lstm_layer)
#
#     # fusion
#     # flatten=concatenate([flatten1,flatten2])
#     #  flatten=Dropout(0.1)(flatten2)
#     #  dense = Dense(728, activation='relu')(flatten)
#     dense = Lambda(outer_product, name='outer_product')([lstm_layer1, lstm_layer2])
#     output = Dense(classes, activation='softmax')(dense)
#     model = Model(model.input, output)
#     model.summary()
#     return model
#
#
# def custom_mobilenetv2_(model, classes):
#     invert1 = model.get_layer('block_2_add').output
#     invert2 = model.get_layer('block_4_add').output
#     invert3 = model.get_layer('block_14_add').output
#     invert4 = model.get_layer('block_15_add').output
#     invert5 = model.get_layer('out_relu').output
#     # reduce the dimension of tensors having 1280 channel
#     invert5 = Conv2D(160, (1, 1), activation='relu')(invert5)
#
#     # SE block
#     invert1_ = Modules.spatial_squeeze_excite_block(invert1)
#     invert2_ = Modules.spatial_squeeze_excite_block(invert2)
#     invert3_ = Modules.spatial_squeeze_excite_block(invert3)
#     invert4_ = Modules.spatial_squeeze_excite_block(invert4)
#     invert5_ = Modules.spatial_squeeze_excite_block(invert5)
#
#     # pool
#     invert1__ = GlobalAveragePooling2D()(invert1_)
#     invert2__ = GlobalAveragePooling2D()(invert2_)
#     invert3__ = GlobalAveragePooling2D()(invert3_)
#     invert4__ = GlobalAveragePooling2D()(invert4_)
#     invert5__ = GlobalAveragePooling2D()(invert5_)
#
#     # pool the layer
#     invert1_ = GlobalAveragePooling2D()(invert1)
#     invert2_ = GlobalAveragePooling2D()(invert2)
#     invert3_ = GlobalAveragePooling2D()(invert3)
#     invert4_ = GlobalAveragePooling2D()(invert4)
#     invert5_ = GlobalAveragePooling2D()(invert5)
#
#     # combine all of them
#     comb = concatenate([
#         # invert1_,
#         # invert2_,
#         invert3_,
#         invert4_,
#         invert5_,
#         # invert1__,
#         # invert2__,
#         invert3__,
#         invert4__,
#         invert5__
#     ])
#     # comb=BatchNormalization()(comb) # added to normalize
#     dense = Dense(1024, activation='relu')(comb)  # reduced the 1024->768 (93.19% accuracy), 1024->1024 (93.50%)
#     dense = Dense(1024, activation='relu')(dense)
#     # softmax
#     output = Dense(classes, activation='softmax')(dense)
#     model = Model(inputs=model.input, outputs=output)
#     return model

if __name__ == '__main__':
    # model = MobileNetv2_((224, 224, 3), 30, 1.0)
    # modify the mobilenetv2 by adding the SEblock
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    model.summary()
    model.trainable = True
    #
    # # ##batch normalization
    # for layer in model.layers:
    #     if isinstance(layer, BatchNormalization):
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True

    # inputs = Input(shape=(224, 224, 3))
    # model = m(inputs, training=False)  # because of BN
    # print(model.summary)
    # disable the training of pre-trained model
    # for layer in model.layers:
    #     layer.trainable = True

    # model=fine_tune(model,30)
    # model.summary()
    # model=custom_mobilenetv2(model, 30)
    # print(model.summary())
    acc = []
    for i in range(1, 5):
        # data load and train
        root_path = "//ad.monash.edu/home/User066/csit0004/Desktop/Jagannath_dai/AID_/5_5/" + str(i + 1) + '/'
        DATASET_PATH = root_path + 'train'
        test_dir = root_path + 'val'
        # DATASET_PATH = root_path/train'
        # test_dir = 'root_path/val'
        IMAGE_SIZE = (224, 224)
        data_list = os.listdir(DATASET_PATH)
        # data_list = os.listdir('D:/COVID/four_classes/splits/f4/train')
        # Delete some classes that may interfere
        print(len(data_list))
        NUM_CLASSES = len(data_list)
        BATCH_SIZE = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
        NUM_EPOCHS = 50
        LEARNING_RATE = 0.0003

        # Train datagen here is a preprocessor
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           # rotation_range=50,
                                           # width_shift_range=0.2,
                                           # height_shift_range=0.2,
                                           # shear_range=0.25,
                                           # zoom_range=0.1,
                                           # channel_shift_range=20,
                                           horizontal_flip=True,
                                           # ertical_flip=True,
                                           # validation_split=0.2,
                                           # fill_mode='constant'
                                           )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # For multiclass use categorical n for binary us
        train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                          target_size=IMAGE_SIZE,
                                                          shuffle=True,
                                                          interpolation='lanczos:random',  # random crop
                                                          batch_size=BATCH_SIZE,
                                                          # subset="training",
                                                          seed=42,
                                                          class_mode="categorical"
                                                          # For multiclass use categorical n for binary use binary
                                                          )

        valid_batches = test_datagen.flow_from_directory(test_dir,
                                                         target_size=IMAGE_SIZE,
                                                         shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         # subset="validation",
                                                         seed=42,
                                                         class_mode="categorical"
                                                         # For multiclass use categorical n for binary use binary
                                                         )

        # fine-tune the mobilenetv2 first and then perform the feature extraction as suggested
        # base_model = model.output
        # gap = GlobalAveragePooling2D()(base_model)
        # output = Dense(30, activation='softmax')(gap)
        # model_ = Model(model.input, output)
        #
        # # compile and train the model
        #
        # model_.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
        #                # optimizer=optimizers.SGD(lr=LEARNING_RATE,momentum=0.9),
        #                optimizer=optimizers.Adam(lr=LEARNING_RATE, decay=1e-05),
        #                #  optimizer=optimizers.Adam(lr_schedule),
        #                metrics=['acc'])
        #
        # # print(model.summary())
        # # learning schedule callback
        # es = EarlyStopping(monitor='val_loss', patience=5)
        # lrate = LearningRateScheduler(step_decay)
        # callbacks_list = [lrate]
        #
        # STEP_SIZE_TRAIN = train_batches.n // train_batches.batch_size
        # STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size
        # # lr_decay = LearningRateScheduler(schedule=lambda epoch: LEARNING_RATE * (0.9 ** epoch))
        # # callbacks_list=[lr_decay]
        # result = model_.fit_generator(train_batches,
        #                               steps_per_epoch=STEP_SIZE_TRAIN,
        #                               validation_data=valid_batches,
        #                               validation_steps=STEP_SIZE_VALID,
        #                               epochs=NUM_EPOCHS,
        #                               callbacks=callbacks_list
        #                               )

        # perform the training based on the fine-tuned model
        # m = custom_mobilenetv2(model, 30)
        # m = gru_lstm_mobilenetv2(model, 30)
        # m = fine_tune(model, 30)
        m = custom_resnet50(model, 30)
        print(m.summary())
        # m=model

        m.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
                  # optimizer=optimizers.SGD(lr=LEARNING_RATE,momentum=0.9),
                  optimizer=optimizers.Adam(lr=LEARNING_RATE, decay=1e-05),
                  #  optimizer=optimizers.Adam(lr_schedule),
                  metrics=['acc'])

        # print(model.summary())
        # learning schedule callback
        es = EarlyStopping(monitor='val_loss', patience=5)
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]

        STEP_SIZE_TRAIN = train_batches.n // train_batches.batch_size
        STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size
        # lr_decay = LearningRateScheduler(schedule=lambda epoch: LEARNING_RATE * (0.9 ** epoch))
        # callbacks_list=[lr_decay]
        result = m.fit_generator(train_batches,
                                 steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=valid_batches,
                                 validation_steps=STEP_SIZE_VALID,
                                 epochs=NUM_EPOCHS,
                                 callbacks=callbacks_list
                                 )

        #  print('Training time:' + str(time.clock() - train_s_time) + 'secs.')

        # Firstly train the vgg-16 model and perform attention module operations

        # import matplotlib
        #
        # matplotlib.use('TKAgg')
        # import matplotlib.pyplot as plt
        #
        # print(result.history.keys())
        # fig1 = plt.figure(1)
        # # summarize history for accuracy
        # plt.plot(result.history['acc'])
        # plt.plot(result.history['val_acc'])
        # plt.title('Model Accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Training', 'Validation'], loc='upper left')
        # #plt.savefig(root_path + '/' + 'acc.png')
        # plt.show()
        #
        # # summarize history for loss
        # fig2 = plt.figure(2)
        # plt.plot(result.history['loss'])
        # plt.plot(result.history['val_loss'])
        # plt.title('Model Loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Training', 'Validation'], loc='upper left')
        # #plt.savefig(root_path + '/' + 'loss.png')
        # plt.show()

        # test_datagen = ImageDataGenerator(rescale=1. / 255)
        # eval_generator = test_datagen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=1,
        #                                                   shuffle=True, seed=42, class_mode="categorical")
        # eval_generator.reset()
        # eval_generator.reset()
        # test_s_time = time.clock()

        x = m.evaluate_generator(valid_batches,
                                 steps=np.ceil(len(valid_batches)),
                                 use_multiprocessing=False,
                                 verbose=1,
                                 workers=1,
                                 )
        #  print('Testing time:' + str(time.clock() - test_s_time) + 'secs.')
        print('Test loss:', x[0])
        print('Test accuracy:', x[1])
        acc.append(x[1])
        # release memory
        K.clear_session()
        # del model
        del m
        break;

    # print the accuracy
    print(acc)
    a = np.array(acc)
    print('The averaged accuracy is:\n')
    print(np.mean(a))
    print('The std is: \n')
    print(np.std(a))
