"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""
import Modules
import numpy as np
# import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, Conv2D, add, TimeDistributed, GlobalAveragePooling2D, \
    Dropout, \
    Concatenate, concatenate, \
    Dense, GlobalMaxPooling2D, MaxPooling2D, Lambda, Add
# from tensorflow.python.keras.layers import Activation, BatchNormalization, Lambda, Add, LSTM, GRU, Reshape, \
#     Concatenate, \
#     Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
import keras_preprocessing
import preprocess_crop
import os, math  # , time
import tensorflow as tf
import cbam_attention
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# learning decay rate schedule

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.4  # in 0.5 it provided an accuracy of 80%+
    epochs_drop = 4.0  # 5.0 gives an optimal epochs_drop
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def weighted_pooling(invert2):
    alpha = 0.7
    a = GlobalAveragePooling2D()(invert2)
    a = Lambda(lambda xx: xx * alpha)(a)
    m = GlobalMaxPooling2D()(invert2)
    m = Lambda(lambda xx: xx * (1 - alpha))(m)
    x3 = Add()([a, m])
    return x3


def custom_resnet50(model, classes):
    invert11_ = model.get_layer('conv2_block3_out').output
    invert1 = model.get_layer('conv3_block4_out').output
    invert2 = model.get_layer('conv4_block6_out').output
    invert3 = model.get_layer('conv5_block3_out').output

    # EARCM module around 92.4%
    invert11_ = Modules.EACRM(invert11_)
    invert1 = Modules.EACRM(invert1)
    invert2 = Modules.EACRM(invert2)
    invert3 = Modules.EACRM(invert3)

    # cbam-test around 92%
    # invert11_ = cbam_attention.cbam_block(invert11)
    # invert1 = cbam_attention.cbam_block(invert1)
    # invert2 = cbam_attention.cbam_block(invert2)
    # invert3 = cbam_attention.cbam_block(invert3)

    # improved cbam: around 90%
    # invert11_ = cbam_attention.cbam_block_improved(invert11)
    # invert1 = cbam_attention.cbam_block_improved(invert1)
    # invert2 = cbam_attention.cbam_block_improved(invert2)
    # invert3 = cbam_attention.cbam_block_improved(invert3)

    # pool the layer using GAP
    invert11_ = GlobalAveragePooling2D()(invert11_)
    invert1_ = GlobalAveragePooling2D()(invert1)
    invert2_ = GlobalAveragePooling2D()(invert2)
    invert3_ = GlobalAveragePooling2D()(invert3)

    # weighted pooling
    # invert11_ = weighted_pooling(invert11_)
    # invert1_ = weighted_pooling(invert1)
    # invert2_ = weighted_pooling(invert2)
    # invert3_ = weighted_pooling(invert3)

    # combine all of them
    comb = add([
        invert11_,
       # inv11,
        invert1_,
       # inv1,
        invert2_,
       # inv2,
        invert3_,
       # inv3
    ])
    # comb = BatchNormalization()(comb)  # added to normalize
    dense = Dense(1024, activation='relu')(comb)  # reduced the 1024->768
    dense = Dense(768, activation='relu')(dense)  # reduced the 1024->768
    # softmax
    output = Dense(classes, activation='softmax')(dense)
    model = Model(inputs=model.input, outputs=output)
    return model


def train_ml(model, train_batches, valid_batches, classes,NUM_EPOCHS):
    m = custom_resnet50(model, classes)
    print(m.summary())
    # m=model
    #NUM_EPOCHS = 30
    LEARNING_RATE = 0.0003
    m.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
              # optimizer=optimizers.SGD(lr=LEARNING_RATE,momentum=0.9),
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              #  optimizer=optimizers.Adam(lr_schedule),
              metrics=['acc'])

    # print(model.summary())
    # learning schedule callback
    # es = EarlyStopping(monitor='val_loss', patience=5)
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    STEP_SIZE_TRAIN = train_batches.n // train_batches.batch_size
    STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size
    # lr_decay = LearningRateScheduler(schedule=lambda epoch: LEARNING_RATE * (0.9 ** epoch))
    # callbacks_list=[es]
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
    return x, m


# def custom_resnet50_(model, classes):
#     alpha = 0.7
#     invert11 = model.get_layer('conv2_block3_out').output
#     invert1 = model.get_layer('conv3_block4_out').output
#     invert2 = model.get_layer('conv4_block6_out').output
#     invert3 = model.get_layer('conv5_block3_out').output
#
#     # pool the layer
#
#     # invert11 = cbam_attention.channel_attention(invert11)
#     a = GlobalAveragePooling2D()(invert11)
#     a = Lambda(lambda xx: xx * alpha)(a)
#     m = GlobalMaxPooling2D()(invert11)
#     m = Lambda(lambda xx: xx * (1 - alpha))(m)
#     x1 = Add()([a, m])
#     # x1=GlobalAveragePooling2D()(invert11)
#
#     # invert11_ = GlobalAveragePooling2D()(invert11)
#     # invert1 = cbam_attention.cbam_block(invert1)
#     # invert1=squeeze_excite_block(invert1)
#     # invert1 = Modules.spatial_attention(invert1)
#     a = GlobalAveragePooling2D()(invert1)
#     a = Lambda(lambda xx: xx * alpha)(a)
#     m = GlobalMaxPooling2D()(invert1)
#     m = Lambda(lambda xx: xx * (1 - alpha))(m)
#     x2 = Add()([a, m])
#
#     # invert1_ = GlobalAveragePooling2D()(invert1)
#     # invert2 = cbam_attention.cbam_block(invert2)
#     # invert2 = squeeze_excite_block(invert2)
#     # invert2 = Modules.spatial_attention(invert2)
#     a = GlobalAveragePooling2D()(invert2)
#     a = Lambda(lambda xx: xx * alpha)(a)
#     m = GlobalMaxPooling2D()(invert2)
#     m = Lambda(lambda xx: xx * (1 - alpha))(m)
#     x3 = Add()([a, m])
#
#     # invert2_ = GlobalAveragePooling2D()(invert2)
#     # invert3 = cbam_attention.cbam_block(invert3)
#     # invert3 = Modules.EACRM(invert3)
#     a = GlobalAveragePooling2D()(invert3)
#     a = Lambda(lambda xx: xx * alpha)(a)
#     m = GlobalMaxPooling2D()(invert3)
#     m = Lambda(lambda xx: xx * (1 - alpha))(m)
#     x4 = Add()([a, m])
#
#     # combine all of them
#     comb = concatenate([
#         # invert11_,
#         # invert1_,
#         # invert2_,
#         # invert3_,
#         x1,
#         x2,
#         x3,
#         x4
#     ])
#
#     # comb=BatchNormalization()(comb)
#     dense = Dense(1024, activation='relu')(comb)  # reduced the 1024->768
#     dense = Dense(768, activation='relu')(dense)  # reduced the 1024->768
#     # softmax
#     output = Dense(classes, activation='softmax')(dense)
#     model = Model(inputs=model.input, outputs=output)
#     return model


if __name__ == '__main__':
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # model =ResNet50()
    model.summary()
    model.trainable = True
    # for layer in model.layers[:20]:  # upto 36 is alright
    #     layer.trainable = False
    # j=0
    # for i in model.layers:
    #     print(str(j)+str(i.name))
    #     j=j+1
    acc = []
    for i in range(0, 20):
        # # data load and train
        # root_path = "D://Jagannath_dai/NWPU_/1_9/" + str(1 + 1) + '/'
        # DATASET_PATH = root_path + 'train'
        # test_dir = root_path + 'val'
        # # DATASET_PATH = root_path/train'
        # # test_dir = 'root_path/val'
        # IMAGE_SIZE = (224, 224)
        # data_list = os.listdir(DATASET_PATH)
        # # data_list = os.listdir('D:/COVID/four_classes/splits/f4/train')
        # # Delete some classes that may interfere
        # print(len(data_list))
        # NUM_CLASSES = len(data_list)
        # BATCH_SIZE = 16  # try reducing batch size or freeze more layers if your GPU runs out of memory
        #
        # # Train datagen here is a preprocessor
        # train_datagen = ImageDataGenerator(rescale=1. / 255,
        #                                    # rotation_range=50,
        #                                    # width_shift_range=0.2,
        #                                    # height_shift_range=0.2,
        #                                    # shear_range=0.25,
        #                                    # zoom_range=0.1,
        #                                    # channel_shift_range=20,
        #                                    horizontal_flip=True,
        #                                    # ertical_flip=True,
        #                                    # validation_split=0.2,
        #                                    # fill_mode='constant'
        #                                    )
        #
        # test_datagen = ImageDataGenerator(rescale=1. / 255)
        #
        # train_batches = train_datagen.flow_from_directory(DATASET_PATH,
        #                                                   target_size=IMAGE_SIZE,
        #                                                   shuffle=True,
        #                                                   interpolation='lanczos:random',  # <--------- random crop
        #                                                   batch_size=BATCH_SIZE,
        #                                                   # subset="training",
        #                                                   seed=42,
        #                                                   class_mode="categorical"
        #                                                   # For multiclass use categorical n for binary use binary
        #                                                   )
        #
        # valid_batches = test_datagen.flow_from_directory(test_dir,
        #                                                  target_size=IMAGE_SIZE,
        #                                                  shuffle=True,
        #                                                  batch_size=BATCH_SIZE,
        #                                                  # interpolation = 'lanczos:center', # <--------- center crop
        #                                                  # subset="validation",
        #                                                  seed=42,
        #                                                  class_mode="categorical"
        #                                                  # For multiclass use categorical n for binary use binary
        #                                                  )
        # x,m = train_ml(model, train_batches, valid_batches, NUM_CLASSES,30)
        # print('Test loss:', x[0])
        # print('Test accuracy:', x[1])
        # # acc.append(x[1])
        #
        # # Cross transfer learning approach from the NWPU dataset
        # print('*'*100)
        # print('Starting cross-transfer learning')
        # # m.trainable=False
        # # # new small model
        # output=m.get_layer('dense_17').output
        # output= Dense(30,activation='softmax')(output)
        # model= Model(inputs=m.inputs, outputs=output)

        #root_path = "D://Jagannath_dai/AID_/2_8/" + str(i + 1) + '/'
        root_path="C:/Users/csitaula/Desktop/Aerial/AID_/2_8/"+str(i+1)+'/'
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
        BATCH_SIZE = 16  # try reducing batch size or freeze more layers if your GPU runs out of memory

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

        train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                          target_size=IMAGE_SIZE,
                                                          shuffle=True,
                                                          interpolation='lanczos:random',  # <--------- random crop
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
                                                         # interpolation = 'lanczos:center', # <--------- center crop
                                                         # subset="validation",
                                                         seed=42,
                                                         class_mode="categorical"
                                                         # For multiclass use categorical n for binary use binary
                                                         )
        x = train_ml(model, train_batches, valid_batches, NUM_CLASSES,30)
        print('Test loss:', x[0])
        print('Test accuracy:', x[1])
        acc.append(x[1])

    # print the accuracy
    print(acc)
    a = np.array(acc)
    print('The averaged accuracy is:\n')
    print(np.mean(a))
    print('The std is: \n')
    print(np.std(a))
