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
    Dense, GlobalMaxPooling2D, MaxPooling2D
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


import tensorflow as tf

def custom_resnet50(model, classes):

    invert11 = model.get_layer('conv2_block3_out').output
    invert1 = model.get_layer('conv3_block4_out').output
    invert2 = model.get_layer('conv4_block6_out').output
    invert3 = model.get_layer('conv5_block3_out').output

    #  pool
    invert11_ = GlobalAveragePooling2D()(invert11)
    invert1_ = Modules.EACRM(invert1)
    invert2_ = Modules.EACRM(invert2)
    invert3_ = Modules.EACRM(invert3)

    # pool the layer
    # invert11_ = GlobalAveragePooling2D()(invert11_)
    invert1_ = GlobalAveragePooling2D()(invert1_)
    invert2_ = GlobalAveragePooling2D()(invert2_)
    invert3_ = GlobalAveragePooling2D()(invert3_)

    # combine all of them
    comb = concatenate([
        invert11_,
        invert1_,
        invert2_,
        invert3_
    ])
    # comb = BatchNormalization()(comb)  # added to normalize
    dense = Dense(512, activation='relu')(comb)  # reduced the 1024->768
    # softmax
    output = Dense(classes, activation='softmax')(dense)
    model = Model(inputs=model.input, outputs=output)
    return model


if __name__ == '__main__':
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # model =ResNet50()
    model.summary()
    # model.trainable = True

    for layer in model.layers[:36]:
        layer.trainable = False

    j=0
    for i in model.layers:
        print(str(j)+str(i.name))
        j=j+1

    acc = []
    for i in range(1, 5):
        # data load and train
        root_path = "D://Jagannath_dai/AID_/2_8/" + str(i + 1) + '/'
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

        train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                          target_size=IMAGE_SIZE,
                                                          shuffle=True,
                                                          interpolation='lanczos',  # random crop
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
                                                         interpolation='lanczos',  # random crop
                                                         # subset="validation",
                                                         seed=42,
                                                         class_mode="categorical"
                                                         # For multiclass use categorical n for binary use binary
                                                         )
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
