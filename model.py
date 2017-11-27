import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, RepeatVector, Merge, Activation
from keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM
from keras.layers import TimeDistributed, Dense, Input, Flatten
from keras.applications import ResNet50, VGG16
from keras.optimizers import RMSprop
from keras.layers.merge import Concatenate
from preprocess import MAX_WORDS, OUTDIM_EMB, WordToWordDistance, VOCAB_SIZE
import keras.backend as K

'''
Rough Estimate
=============

Vocab Size = 37737
'''

OutNeuronLang = 128

'''
def build_imodel():
    kernel = (3,3)
    imodel = Sequential()
    imodel.add(Conv2D(32,kernel,activation='relu', input_shape=(320,240,3)))
    imodel.add(Conv2D(32,kernel,activation='relu'))
    imodel.add(MaxPooling2D(pool_size=(2,2)))
    imodel.add(Dropout(0.25))
    imodel.add(Conv2D(64, kernel, activation='relu'))
    imodel.add(Conv2D(64, kernel, activation='relu'))
    imodel.add(MaxPooling2D(pool_size=(2, 2)))
    imodel.add(Dropout(0.25))
    imodel.add(Flatten())
    return imodel
'''

def sentence_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(K.abs(y_true-y_pred)),axis=1,keepdims=True))
        
def model_vgg_tanh_endglove(CAPTION_LEN):
    print "Creating Model with Vocab Size : NONE " #% VOCAB_SIZE
    #assert os.path.exists('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    #np.set_printoptions(threshold=np.nan)
    #print embeddingMatrixRef[0]
    #print np.shape(embeddingMatrixRef[0])
    cmodel  = Sequential()
    #cmodel.add(Embedding(
    #        input_dim = MAX_WORDS,
    #        output_dim = OUTDIM_EMB,
    #        input_length = CAPTION_LEN,
    #        weights = embeddingMatrixRef,
    #        trainable = False
    #        ))
    cmodel.add(LSTM(OUTDIM_EMB, input_shape=(CAPTION_LEN+1,OUTDIM_EMB ), return_sequences=True))#,kernel_initializer='random_normal'))
    cmodel.add(TimeDistributed(Dense(500)))
    cmodel.add(TimeDistributed(Dense(1000)))
    cmodel.add(LSTM(1000, return_sequences = True))
    #cmodel.add(LSTM(OUTDIM_EMB, return_sequences=True))
    #cmodel.add(LSTM(128, return_sequences=False))
    cmodel.summary()
    
    input_tensor = Input(shape=(224,224,3))
    res = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor) #build_imodel() 
    #res.trainable = False
    res.summary()
    imodel = Sequential(layers=res.layers)
    #for i,l in enumerate(res.layers):
    #    print "Adding layer %d" % i
        #l.trainable = False
    #    imodel.add(l)
    for lay in imodel.layers:
        print lay.get_weights()
        lay.trainable = False
    imodel.add(Flatten())
    imodel.add(RepeatVector(CAPTION_LEN + 1))
    
    #imodel.trainable = False
    #for layer in imodel.layers:
    #    layer.trainable = False
    
    imodel.summary()

    model = Sequential()
    model.add(Merge([cmodel,imodel],mode='concat'))
    #model.add(RepeatVector(CAPTION_LEN))
    model.add(LSTM(1000,return_sequences=True))#,kernel_initializer='random_normal'))
    model.add(TimeDistributed(Dense(500)))#,kernel_initializer='random_normal'))
    #model.add(LSTM(512,return_sequences=True))
    model.add(TimeDistributed(Dense(OUTDIM_EMB,kernel_initializer='random_normal')))
    model.add(Activation('tanh'))
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy',sentence_distance])
    model.summary()
    print "Model Created model_vgg_tanh_endglove"
    return model

def model_resnet_endonehot(CAPTION_LEN):
    print "Creating Model with Vocab Size :  %d " % VOCAB_SIZE[0]
    cmodel  = Sequential()
    cmodel.add(LSTM(256, input_shape=(CAPTION_LEN+1,OUTDIM_EMB ), return_sequences=True))
    cmodel.add(TimeDistributed(Dense(512)))
    cmodel.add(TimeDistributed(Dropout(0.2)))
    cmodel.summary()
    
    input_tensor = Input(shape=(224,224,3))
    res = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    res.trainable = False
    imodel = Sequential()
    imodel.add(res)
    imodel.add(Flatten())
    imodel.add(Dense(512,activation='relu'))
    imodel.add(Dropout(0.2))
    imodel.add(RepeatVector(CAPTION_LEN + 1))
    
    imodel.summary()

    model = Sequential()
    model.add(Merge([cmodel,imodel],mode='concat'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(LSTM(1024,return_sequences=True))#,kernel_initializer='random_normal'))
    model.add(TimeDistributed(Dense(VOCAB_SIZE[0],kernel_initializer='random_normal')))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print "Model Created model_resnet_endonehot"
    return model

def build_model(CAPTION_LEN):
    return model_resnet_endonehot(CAPTION_LEN)
