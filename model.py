import cv2
import os, sys
import numpy as np
from logger import logger
import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, RepeatVector, Merge, Activation, dot, Permute, Reshape, Multiply
from keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization
from keras.layers import TimeDistributed, Dense, Input, Flatten, GlobalAveragePooling2D, Bidirectional
from keras.layers import concatenate
from keras.applications import ResNet50, VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.layers.merge import Concatenate
from keras.models import Model
from vocab import Vocab
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from keras import regularizers
from Attention import Attention
from keras import backend as K

def sentence_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(K.abs(y_true-y_pred)),axis=1,keepdims=True))

class VModel:

    def __init__(self, CAPTION_LEN, VOCAB_SIZE, cutoffonly = False, learning = True):
        self.CAPTION_LEN = CAPTION_LEN
        self.VOCAB_SIZE  = VOCAB_SIZE
        if not cutoffonly:
            self.build_mcnn(self.CAPTION_LEN, self.VOCAB_SIZE, learning = learning)
        self.build_cutoffmodel()

    def  get_model(self):
        return self.model

    '''
    Attempt to split pretrained CNN out of model125

    To cache a lower dimension vector per frame to file
    # PC : pretrained CNN will be non-trainable now
    '''
    def build_cutoffmodel(self):
        base = ResNet50(include_top = False, weights='imagenet')
        # base = InceptionV3(include_top = False, weights='imagenet')
        self.co_model = base
        logger.debug("Building Cutoff Model")
        self.co_model.summary()
        self.co_model._make_predict_function()
        self.graph = tf.get_default_graph()
        logger.debug("Building Cutoff Model : Completed")
        return self.co_model

    # co == Cutoff Model
    def co_getoutshape(self, assert_model = None):
        # ResNet
        shape = (40,2048)
        ## Inception V3
        # shape = (None, 8*8*2048)
        logger.debug("Model Cutoff OutShape : %s" % str(shape))
        '''
        # Not in use
        if assert_model is not None:
            ashape = assert_model.output_shape
            sz = 1
            for x in ashape:
                if x is not None:
                    sz = sz * x
            ashape = (None, sz)
            logger.debug("Assert Model Cutoff OutShape : %s" % str(ashape))
            assert shape == ashape
        '''
        assert len(shape) == 2
        #assert shape[0] is None
        return shape

    def preprocess_partialmodel(self, frames):
        frames_in = np.asarray([image.img_to_array(frame) for frame in frames])
        frames_in = preprocess_input(frames_in)
        with self.graph.as_default():
            frames_out = self.co_model.predict(frames_in)
            frames_out = np.array([frame.flatten() for frame in frames_out])
        return frames_out

    def train_mode(self):
        K.set_learning_phase(1)

    def build_mcnn(self, CAPTION_LEN, VOCAB_SIZE, learning = True):
        if learning:
            self.train_mode()
        from VideoDataset.videohandler import VideoHandler
        CMODEL_NODES = 512
        VIDEO_LENGTH = 40
        logger.debug("Creating Model (CNN Cutoff) with Vocab Size :  %d " % VOCAB_SIZE)
        cmodel_inputs  = Input(shape=(CAPTION_LEN+1,Vocab.OUTDIM_EMB),name='words_input')
        intermediate = TimeDistributed(Dense(512,kernel_initializer='random_normal'))(cmodel_inputs)
        cmodel_outputs = LSTM(CMODEL_NODES, return_sequences=True,kernel_initializer='random_normal')(intermediate) # (CAPTION_LEN, 512)
        cmodel = Model(inputs = cmodel_inputs, outputs = cmodel_outputs) 
        cmodel.summary()
    
        input_shape_audio = VideoHandler.AUDIO_FEATURE
        amodel_inputs = Input(shape=input_shape_audio, name='audio_input')
        intermediate = GRU(128,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     return_sequences=True)(amodel_inputs)
        intermediate = BatchNormalization()(intermediate)
        intermediate = GRU(64,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     return_sequences=True)(intermediate)
        intermediate = BatchNormalization()(intermediate)
        intermediate = Flatten()(intermediate)
        amodel_outputs = RepeatVector(CAPTION_LEN + 1)(intermediate)
        amodel = Model(inputs = amodel_inputs, outputs = amodel_outputs) 
        amodel.summary()

        input_shape_vid = self.co_getoutshape()
        imodel_inputs = Input(shape=input_shape_vid, name='video_input')
        intermediate1 = TimeDistributed(Dense(1024,kernel_initializer='random_normal'))(imodel_inputs)
        intermediate2 = TimeDistributed(Dropout(0.20))(intermediate1)
        intermediate3 = TimeDistributed(BatchNormalization(axis=-1))(intermediate2)
        intermediate4 = TimeDistributed(Activation('tanh'))(intermediate3)
        intermediate5_layer = Bidirectional(GRU(1024, return_sequences=True, kernel_initializer='random_normal'))
        intermediate5 = intermediate5_layer(intermediate4)
        intermediate6 = Flatten()(intermediate5)
        intermediate7 = RepeatVector(CAPTION_LEN + 1)(intermediate6) # (CAPTION_LEN, VIDEO_LENGTH * FEATURE_LEN)
        intermediate8 = Reshape((CAPTION_LEN+1, VIDEO_LENGTH, 2048))(intermediate7)
        
        imodel = Model(inputs = imodel_inputs, outputs = intermediate8)          
        imodel.summary()

        # For every ith Word Generation
        # Give weights to each frame 
        intermediate_vatt = Flatten()(intermediate5) # (VIDEO_LENGTH * 1024)
        intermediate_vatt = RepeatVector(CAPTION_LEN + 1)(intermediate_vatt) # (CAPTION_LEN, VIDEO_LENGTH * 1024)
        intermediate_vatt = Reshape((CAPTION_LEN + 1, VIDEO_LENGTH, 2048))(intermediate_vatt) # (CAPTION_LEN, VIDEO_LENGTH, 1024)

        intermediate_catt = Flatten()(cmodel_outputs)
        intermediate_catt = RepeatVector(VIDEO_LENGTH)(intermediate_catt)
      
        intermediate_catt = Reshape((VIDEO_LENGTH, CAPTION_LEN+1, CMODEL_NODES))(intermediate_catt)
        intermediate_catt = Permute((2,1,3))(intermediate_catt)

        intermediate_att = Concatenate()([intermediate_vatt, intermediate_catt]) # (CAPTION_LEN, VIDEO_LENGTH, CMODEL_NODES + 1024)
        intermediate_att = TimeDistributed(TimeDistributed(
                Dense(1#,
                      #kernel_regularizer=regularizers.l2(0.01),
                      #activity_regularizer=regularizers.l1(0.01)
                      )
                ))(intermediate_att) # (CAPTION_LEN, VIDEO_LENGTH, 1)
        intermediate_att = Flatten()(intermediate_att)
        intermediate_att = Reshape((CAPTION_LEN+1, VIDEO_LENGTH))(intermediate_att)
        intermediate_att = TimeDistributed(Activation('softmax'))(intermediate_att) # (CAPTION_LEN, VIDEO_LENGTH)
        intermediate_att = Flatten()(intermediate_att)
        intermediate_att = RepeatVector(2048)(intermediate_att)
        intermediate_att = Reshape((2048,CAPTION_LEN+1,VIDEO_LENGTH))(intermediate_att) # (1024, CAPTION_LEN, VIDEO_LENGTH)
        intermediate_att = Permute((2,3,1))(intermediate_att) # (CAPTION_LEN, VIDEO_LENGTH, 1024)
        intermediate_att = Multiply()([intermediate_att,intermediate8])
        intermediate_att = Permute((1,3,2))(intermediate_att) # (CAPTION_LEN, 1024, VIDEO_LENGTH)

        sum_layer = Dense(1, input_shape=(VIDEO_LENGTH,), trainable=False, name='dense_attention')
        context = sum_layer(intermediate_att)
        sum_layer_weights = sum_layer.get_weights()
        sum_layer_weights[0].fill(1)
        sum_layer_weights[1].fill(0)
        sum_layer.set_weights(sum_layer_weights)
        context = Reshape((CAPTION_LEN+1,2048))(context)
      
        model_inputs = concatenate([cmodel_outputs, amodel_outputs, context])
        intermediate = TimeDistributed(Dropout(0.2))(model_inputs)
        intermediate = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01))(intermediate)
        intermediate = TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal'))(intermediate)
        model_outputs = TimeDistributed(Activation('softmax'))(intermediate)
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
        model = Model(inputs = [cmodel_inputs, amodel_inputs, imodel_inputs], outputs = model_outputs, name='model')
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.debug("Model Created CAttention_ResNet_D512L512_G128G64_D1024D0.25BN_BDGRU1024_D0.2L1024DVS")
        self.model = model
        return model

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.model, to_file=filename, show_shapes = True, show_layer_names = False)
        print("Model Plotted in %s"%filename)

if __name__ == "__main__":
    if sys.argv[1] == "plot_model":
        from vocab import Vocab
        vmodel = VModel(Vocab.CAPTION_LEN, Vocab.VOCAB_SIZE)
        vmodel.plot_model(sys.argv[2])

