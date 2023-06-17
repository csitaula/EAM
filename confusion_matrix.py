# Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sn
import Modules
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, math  # , time
import tensorflow as tf
from tensorflow.keras.models import load_model
import atrouspp
from aspp import ConvBlock

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)



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
import atrouspp

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
    # invert11_ = GlobalAveragePooling2D()(invert11_)
    # invert1_ = GlobalAveragePooling2D()(invert1)
    # invert2_ = GlobalAveragePooling2D()(invert2)
    # invert3_ = GlobalAveragePooling2D()(invert3)

    # Dimension reduction
    # invert11_= Modules.multi_scale(invert11_)
    # invert1= Modules.multi_scale(invert1)
    # invert2= Modules.multi_scale(invert2)
    # invert3= Modules.multi_scale(invert3)

    # ASPP
    invert11_ = atrouspp.AtrousSpatialPyramidPooling(invert11_)
    invert1 = atrouspp.AtrousSpatialPyramidPooling(invert1)
    invert2 = atrouspp.AtrousSpatialPyramidPooling(invert2)
    invert3 = atrouspp.AtrousSpatialPyramidPooling(invert3)

    # GAP
    invert11_ = GlobalAveragePooling2D()(invert11_)
    invert1_ = GlobalAveragePooling2D()(invert1)
    invert2_ = GlobalAveragePooling2D()(invert2)
    invert3_ = GlobalAveragePooling2D()(invert3)

    # combine all of them
    comb = concatenate([
        invert11_,
        invert1_,
        invert2_,
        invert3_
    ])
    # comb = BatchNormalization()(comb)  # added to normalize
    dense = Dense(1024, activation='relu')(comb)  # reduced the 1024->768
    dense = Dense(768, activation='relu')(dense)  # reduced the 1024->768
    # softmax
    output = Dense(classes, activation='softmax')(dense)
    model = Model(inputs=model.input, outputs=output)
    return model


def train_ml(model, train_batches, valid_batches, classes):
    m = custom_resnet50(model, classes)
    print(m.summary())
    # m=model
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0003
    m.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
              # optimizer=optimizers.SGD(lr=LEARNING_RATE,momentum=0.9),
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              #  optimizer=optimizers.Adam(lr_schedule),
              metrics=['acc'])

    # print(model.summary())
    # learning schedule callback
    es = EarlyStopping(monitor='val_loss', patience=5)
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [es,lrate]

    STEP_SIZE_TRAIN = train_batches.n // train_batches.batch_size
    STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size
    # lr_decay = LearningRateScheduler(schedule=lambda epoch: LEARNING_RATE * (0.9 ** epoch))
    # callbacks_list=[es]
    m.fit_generator(train_batches,
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
    # m.save('trained_model_NWPU_fold1_2_8_GRSL1.h5')
    return x, m


if __name__ == '__main__':
    print('Fold:' + str(0 + 1))
    print("*" * 100)
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    model.trainable = True
    # data load and train
    # root_path = "D://Jagannath_dai/AID_/2_8/" + str(i + 1) + '/'
    root_path = "C:/Users/csitaula/Desktop/onedrive/Aerial/AID_/2_8/" + str(0 + 1) + '/'
    DATASET_PATH = root_path + 'train'
    test_dir = root_path + 'val'
    # # DATASET_PATH = root_path/train'
    # # test_dir = 'root_path/val'
    IMAGE_SIZE = (224, 224)
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
    # x,m = train_ml(model, train_batches, valid_batches, NUM_CLASSES)
    # print('Test loss:', x[0])
    # print('Test accuracy:', x[1])

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    valid_batches = test_datagen.flow_from_directory(test_dir,
                                                     target_size=IMAGE_SIZE,
                                                     shuffle=False,
                                                     batch_size=1,
                                                     seed=42,
                                                     class_mode="categorical"
                                                     # For multiclass use categorical n for binary use binary
                                                     )

    valid_batches.reset()
    valid_batches.reset()

    m= load_model('trained_model_NWPU_fold1_2_8_GRSL1.h5',custom_objects={'ConvBlock':ConvBlock})

    test_dir = 'C:/Users/csitaula/Desktop/onedrive/Aerial/NWPU_/2_8/1/val/'

    data_list = os.listdir(test_dir)

    filenames = valid_batches.filenames
    nb_samples = len(filenames)
    # valid_batches.reset()

    print('Prediction starts')
    prdict = m.predict_generator(valid_batches, steps=np.ceil(len(valid_batches)))

    predict = np.argmax(prdict, axis=1)

    classes = valid_batches.classes[valid_batches.index_array]

    # for acc
    acc = sum(predict == classes) / len(predict)
    print(acc)

    names = data_list

    print(classification_report(valid_batches.classes, predict))

    report = classification_report(classes, predict, output_dict=True, labels=names)

    df = pd.DataFrame(report).transpose()

    # Write it into csv format
    df.to_csv('prf.csv', index=True, header=True)
    data = confusion_matrix(classes, predict)

    # normalise
    data = data.astype('float') / data.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(data, columns=np.unique(names), index=np.unique(names))

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=(25, 25))
    ax = plt.subplot(222)

    sn.set(font_scale=0.4)  # for label size
    h_m = sn.heatmap(df_cm, cmap="icefire",
                     annot=True, annot_kws={"size": 10, "fontweight": "bold",
                                                                "color": "black"},  # 'd',
                     cbar=False,fmt='.1f')  # font size
    # h_m.set(font_scale=10)
    h_m.set_yticklabels(h_m.get_yticklabels(), rotation=0, fontweight="bold",fontsize=10)
    h_m.set_xticklabels(h_m.get_xticklabels(), rotation=90, fontweight="bold",fontsize=10)
    plt.xlabel('Predicted', fontsize=10,fontweight="bold")
    plt.ylabel('Actual', fontsize=10,fontweight="bold")
    plt.show(block=False)
    plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
