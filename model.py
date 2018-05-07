import cv2
import os, sys
import numpy as np
from logger import logger
import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, RepeatVector, Merge, Activation, dot, Permute, Reshape
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
from Attention import Attention
import keras.backend as K

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
        intermediate6 = Attention()(intermediate5)
        #intermediate6 = Flatten()(intermediate5)
        imodel_outputs = RepeatVector(CAPTION_LEN + 1)(intermediate6) # (CAPTION_LEN, VIDEO_LENGTH * FEATURE_LEN)
        context = imodel_outputs
        #imodel_outputs = Reshape((CAPTION_LEN+1, intermediate5_layer.output_shape[1] , intermediate5_layer.output_shape[2] ))(imodel_outputs)
        #print keras.backend.shape(imodel_outputs)

        #imodel = Model(inputs = imodel_inputs, outputs = imodel_outputs)          
        #imodel.summary()

        #intermediate_trans_cmodel_outputs = Permute((2,1))(cmodel_outputs) # (512, CAPTION_LEN)
        #intermediate_trans_cmodel_outputs = Flatten()(intermediate_trans_cmodel_outputs)
        #print keras.backend.shape(intermediate_trans_cmodel_outputs)
        # For (VIDEO_LENGTH, 512 * CAPTION_LEN) shape
        #intermediate_trans_cmodel_outputs = RepeatVector(intermediate5_layer.output_shape[1])(intermediate_trans_cmodel_outputs)
        #print keras.backend.shape(intermediate_trans_cmodel_outputs)
        # For (VIDEO_LENGTH, 512, CAPTION_LEN) shape
        #imodel_outputs = Reshape((intermediate5_layer.output_shape[1], CMODEL_NODES, CAPTION_LEN+1 ))(imodel_outputs)
        # For (CAPTION_LEN, VIDEO_LENGTH, 512) shape
        #intermediate_trans_cmodel_outputs = Permute((3,1,2))(intermediate_trans_cmodel_outputs)
        # For (CAPTION_LEN, VIDEO_LENGTH, 512+1024) shape
        #attention_imodel_cmodel_outputs = Concatenate()([intermediate_trans_cmodel_outputs, cmodel_outputs])
        # For (CAPTION_LEN, VIDEO_LENGTH, 1) shape
        #attention_weights = TimeDistributed(TimeDistributed(Dense(1)))(attention_imodel_cmodel_outputs)
        # For (CAPTION_LEN, VIDEO_LENGTH) shape
        #attention_weights = TimeDistributed(Flatten())(attention_weights)
        
        #attention_weights = dot([cmodel_outputs, intermediate5], axes=[2,2])
        #attention_weights = TimeDistributed(Activation('softmax'))(attention_weights)
        #context = TimeDistributed(Dot(axes=[2,2]))([attention_weights, imodel_outputs])
     
        model_inputs = concatenate([cmodel_outputs, amodel_outputs, context])
        intermediate = TimeDistributed(Dropout(0.2))(model_inputs)
        intermediate = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01))(intermediate)
        intermediate = TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal'))(intermediate)
        model_outputs = TimeDistributed(Activation('softmax'))(intermediate)
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
        model = Model(inputs = [cmodel_inputs, amodel_inputs, imodel_inputs], outputs = model_outputs, name='model')
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.debug("Model Created Attention_ResNet_D512L512_G128G64_D1024D0.25BN_BDGRU1024_D0.2L1024DVS")
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

