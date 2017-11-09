import shutil,json
from keras import callbacks
import os, sys
import numpy as np
import preprocess
from preprocess import VOCAB_SIZE, CAPTION_LEN, imageToVec, get_image_fname, captionToVec, v_ind2word, v_word2ind
from model import build_model
from preprocess import build_vocab, build_dataset, ENG_SOS, ENG_EOS, onehot, word2embd, embdToWord
MFNAME='model_vgg_we_5.5_5.5.dat'
STATE='state_vgg_we_5.5_5.5.txt'
state = {'epochs':1000,'batch_size':20,'super_batch':100}
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
    model = build_model(preprocess.VOCAB_SIZE, preprocess.CAPTION_LEN)
    loadstate()

def train_model(trainset):
    #print [x for x in train_generator(trainset)]
    #return
    #model.fit_generator(train_generator(trainset),steps_per_epoch=20,epochs=30,verbose=2)
    trainX1 = []
    trainX2 = []
    trainY = []
    i = 0
    print "Arranging Trainset"
    while i < len(trainset):
        capS = [v_word2ind[ENG_SOS]] + trainset[i][0]
        capE = trainset[i][0] + [v_word2ind[ENG_EOS]]
        capE = [word2embd(w) for w in capE]
        image = trainset[i][1]
        trainX1.append( capS )
        trainX2.append( image )
        trainY.append( capE )
        if i==0:
            print "capS %s " % capS
            print "capE %s " % capE

        i+=1
    trainX = [np.array( trainX1 ), np.array( trainX2 )]
    trainY = np.array(trainY)

    print "Attempt to Fit Data"
    inEpochs = 15
    model.fit(x=trainX,y=trainY,batch_size=state['batch_size'],epochs=inEpochs,verbose=2, callbacks=[ModelCallback()])
    print "Model Fit"

def train():
    MX = state['epochs']
    for it in range(MX):
        dataset = build_dataset(state['super_batch'])
        print "Outer Iteration %3d of %3d " % (it+1,MX)
        train_model(dataset[0])
        state['epochs']-=1
        savestate()
        

def predict_model(_id):
    imgVec = imageToVec(_id)
    fname = get_image_fname(_id)
    print "Predicting for Image %s " % fname
    l = 0
    capS = ENG_SOS
    while l < preprocess.CAPTION_LEN:
        cap = captionToVec(capS)
        newCapS = model.predict([np.array([cap]),np.array([imgVec])])[0]
        newWord = newCapS[l]
        print newWord
        #ind = np.argmax(newWord)
        newWord = embdToWord(newWord)
        print "NWord %s " % newWord
        l+=1
        
        if newWord == ENG_EOS:
            break
        capS = "%s %s" % (capS, newWord)
        print newCapS
    print fname
    print "Observed : %s " % capS
    actualC = preprocess.lst[_id]
    print "Actual   : %s " % actualC

    
def predict(_id):
    predict_model(_id)

def run():
    build_vocab()
    loadmodel()
    if len(sys.argv) < 3 or '-predict' != sys.argv[1]:
        train()
    else:
        predict(int(sys.argv[2]))
    
if __name__ == '__main__':
    run()
