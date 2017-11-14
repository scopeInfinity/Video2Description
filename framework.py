import shutil,json
from keras import callbacks
import os, sys
import numpy as np
#import preprocess
from preprocess import CAPTION_LEN, ENG_SOS, ENG_EOS, ENG_NONE, DIR_IMAGES
from preprocess import  embeddingIndexRef, imageToVec, word2embd, embdToWord, captionToVec, get_image_caption, build_vocab, build_dataset, build_gloveVocab, get_image_fname
from model import build_model

MFNAME='model_vgg16_we_5.5_5.5_v1.dat'
STATE='state_vgg16_we_5.5_5.5_v1.txt'
state = {'epochs':1000,'inepochs':20,'batch_size':30,'super_batch':100,'val_batch':20}
model = None


def loadstate():
    global model
    global state
    if os.path.exists(MFNAME):
        model.load_weights(MFNAME)
        print "Weights Loaded"
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
            print "State Loaded"


def savestate():
    global model
    try:
        pass
    finally:
        tname = "_"+MFNAME
        model.save_weights(tname)
        shutil.copy2(tname, MFNAME)
        os.remove(tname)
        print "Weights Saved"
        with open(STATE,'w') as f:
            json.dump(state,f)
            print "State Saved"

class ModelCallback(callbacks.Callback):
 
    def on_epoch_end(self, epoch, logs={}):
        #state['epochs']-=1
        #savestate()
        return

def loadmodel():
    global model
    model = build_model(CAPTION_LEN)
    loadstate()

def prepare_feedkeras(trainset):
    trainX1 = []
    trainX2 = []
    trainY = []
    i = 0
    we_sos = [word2embd(ENG_SOS)]
    we_eos = [word2embd(ENG_EOS)]
    print we_sos
    print we_eos
    print "Arranging Trainset"
    while i < len(trainset):
        capS =  we_sos + trainset[i][0] 
        capE =  trainset[i][0] + we_eos
        image = trainset[i][1]
        #print "%d ; %d " %(len(capS),len(capE))
        trainX1.append( capS )
        trainX2.append( image )
        trainY.append( capE )
        if i==0:
            print "capS %s " % capS
            print "capE %s " % capE
            print "Image %s" % str(image)

        i+=1
    trainX = [np.array( trainX1 ), np.array( trainX2 )]
    trainY = np.array(trainY)
    return (trainX,trainY)

def train_model(trainvalset):
    #print [x for x in train_generator(trainset)]
    #return
    #model.fit_generator(train_generator(trainset),steps_per_epoch=20,epochs=30,verbose=2)
    (trainX,trainY) = prepare_feedkeras(trainvalset[0])
    valset = prepare_feedkeras(trainvalset[1])
    print "Attempt to Fit Data"
    inEpochs = state['inepochs']
    model.fit(x=trainX,y=trainY,batch_size=state['batch_size'],epochs=inEpochs, validation_data=valset, verbose=2, callbacks=[ModelCallback()])
    print "Model Fit"

def train(lst):
    MX = state['epochs']
    for it in range(MX):
        dataset = build_dataset(lst, state['super_batch'],state['val_batch'])
        print "Outer Iteration %3d of %3d " % (it+1,MX)
        train_model(dataset)
        state['epochs']-=1
        savestate()
        
def predict_model(lst,_id):
    imgVec = imageToVec(_id)
    fname = get_image_fname(_id)
    print "Predicting for Image %s " % fname
    l = 0
    capS = ENG_SOS
    while l < CAPTION_LEN:
        cap = captionToVec(capS, addOne=True)
        newCapS = model.predict([np.array([cap]),np.array([imgVec])])[0]
        newWord = newCapS[l]
        print newWord
        #ind = np.argmax(newWord)
        newWord,dis = embdToWord(newWord)
        print "NWord %s\tDistance= %f" % (newWord,dis)
        l+=1
        
        if newWord == ENG_EOS:
            break
        capS = "%s %s" % (capS, newWord)
        print newCapS
    print "eog %s%s" % (DIR_IMAGES,fname)
    print "Observed : %s " % capS
    actualC = lst[_id]
    print "Actual   : %s " % actualC

    
def predict(lst,_id):
    predict_model(lst,_id)

def run():
    build_gloveVocab() 
    lst = build_vocab()
    loadmodel()
    if len(sys.argv) < 3 or '-predict' != sys.argv[1]:
        train(lst)
    else:
        predict(lst,int(sys.argv[2]))
    
if __name__ == '__main__':
    run()
