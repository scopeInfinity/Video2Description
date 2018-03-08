import os
import numpy as np
from logger import logger
from keras.models import Sequential
from keras.layers import Dropout, Flatten, RepeatVector, Merge, Activation
from keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM
from keras.layers import TimeDistributed, Dense, Input, Flatten
from keras.applications import ResNet50, VGG16
from keras.optimizers import RMSprop
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K
from vocab import Vocab
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
OutNeuronLang = 128

def sentence_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(K.abs(y_true-y_pred)),axis=1,keepdims=True))

class VModel:

    def __init__(self, CAPTION_LEN, VOCAB_SIZE):
        self.CAPTION_LEN = CAPTION_LEN
        self.VOCAB_SIZE  = VOCAB_SIZE
        self.build_mcnn(self.CAPTION_LEN, self.VOCAB_SIZE)
        self.build_cutoffmodel()

    def  get_model(self):
        return self.model

    '''
    Attempt to split pretrained CNN out of model
    To cache a lower dimension vector per frame to file
    # PC : pretrained CNN will be non-trainable now
    '''
    def build_cutoffmodel(self):
        base = ResNet50(include_top = False, weights='imagenet')
        self.co_model = base
        logger.debug("Building Cutoff Model")
        self.co_model.summary()
        self.co_model._make_predict_function()
        self.graph = tf.get_default_graph()
        logger.debug("Building Cutoff Model : Completed")
        return self.co_model

    # co == Cutoff Model
    def co_getoutshape(self):
        return (None,2048)

    def preprocess_partialmodel(self, frames):
        frames_in = np.asarray([image.img_to_array(frame) for frame in frames])
        frames_in = preprocess_input(frames_in)
        with self.graph.as_default():
            frames_out = self.co_model.predict(frames_in)
            frames_out = np.array([frame.flatten() for frame in frames_out])
        return frames_out

    def build_mcnn(self, CAPTION_LEN, VOCAB_SIZE):
        logger.debug("Creating Model (CNN Cutoff) with Vocab Size :  %d " % VOCAB_SIZE)
        cmodel  = Sequential()
        cmodel.add(LSTM(256, input_shape=(CAPTION_LEN+1,Vocab.OUTDIM_EMB ), return_sequences=True,kernel_initializer='random_normal'))
        cmodel.add(TimeDistributed(Dropout(0.1)))
        cmodel.add(TimeDistributed(Dense(1024,kernel_initializer='random_normal')))
        cmodel.add(TimeDistributed(Dropout(0.2)))
        cmodel.summary()
    
        input_shape = self.co_getoutshape()
        print input_shape
        imodel = Sequential()
        imodel.add(TimeDistributed(Dense(1024,kernel_initializer='random_normal'), input_shape=input_shape))
        imodel.add(TimeDistributed(Dropout(0.2)))
        imodel.add(LSTM(512, return_sequences=False, kernel_initializer='random_normal'))
        imodel.add(RepeatVector(CAPTION_LEN + 1))
         
        imodel.summary()
     
        model = Sequential()
        model.add(Merge([cmodel,imodel],mode='concat'))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(LSTM(1024,return_sequences=True, kernel_initializer='random_normal'))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal')))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        logger.debug("Model Created model_mcnn")
        self.model = model
        return model


def model_resnet_endonehot(CAPTION_LEN,VOCAB_SIZE):
    logger.debug("Creating Model with Vocab Size :  %d " % VOCAB_SIZE)
    cmodel  = Sequential()
    cmodel.add(LSTM(256, input_shape=(CAPTION_LEN+1,Vocab.OUTDIM_EMB ), return_sequences=True,kernel_initializer='random_normal'))
    cmodel.add(TimeDistributed(Dense(1024,kernel_initializer='random_normal')))
    cmodel.add(TimeDistributed(Dropout(0.2)))
    cmodel.summary()
    
    input_tensor = Input(shape=(224,224,3))
    res = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    res.trainable = False
    imodel = Sequential()
    imodel.add(TimeDistributed(res, input_shape=(None, 224, 224, 3)))
    imodel.add(TimeDistributed(Flatten()))
    imodel.add(Dense(1024,kernel_initializer='random_normal'))
    imodel.add(Dropout(0.2))
    imodel.add(Dense(1024,kernel_initializer='random_normal'))
    imodel.add(Dropout(0.1))
    imodel.add(LSTM(512, return_sequences=False, kernel_initializer='random_normal'))
    imodel.add(RepeatVector(CAPTION_LEN + 1))
    
    imodel.summary()

    model = Sequential()
    model.add(Merge([cmodel,imodel],mode='concat'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(LSTM(1024,return_sequences=True, kernel_initializer='random_normal'))
    model.add(LSTM(1024,return_sequences=True, kernel_initializer='random_normal'))
    model.add(TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal')))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    logger.debug("Model Created model_resnet_endonehot")
    return model

