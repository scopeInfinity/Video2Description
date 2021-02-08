import collections
import json
import os
import numpy as np
import pickle
import random
import re 
import shutil
import sys

from sets import Set

from keras import callbacks
from keras.applications import imagenet_utils 
from keras.preprocessing import image
from keras.preprocessing import sequence

from common.logger import logger

ROOT_DIR = '/home/gagan.cs14/btp'
GITBRANCH = os.popen('git branch | grep "*"').read().split(" ")[1][:-1] 
GITBRANCHPREFIX = "/home/gagan.cs14/btp_"+GITBRANCH+"/"
# Parameters
CAPTION_LEN = 10
MAX_WORDS = 400000
OUTENCODINGGLOVE = False

os.chdir(ROOT_DIR)
BADLOGS = GITBRANCHPREFIX+"badlogs.txt"
FILENAME_CAPTION = 'ImageDataset/annotations/captions_train2014.json'
DIR_IMAGES = 'ImageDataset/train2014/'
DIR_IMAGESP = 'ImageDataset/processed/'
VOCAB_FILE = GITBRANCHPREFIX+"vocab.dat"
GLOVE_FILE = 'glove/glove.6B.100d.txt'
OUTDIM_EMB = 100
USE_GLOVE = True
WORD_MIN_FREQ = 5
def get_image_fname(_id):
    return '%sCOCO_train2014_%012d.jpg' % (DIR_IMAGES, _id)


vocab = Set([])
v_ind2word = {}
v_word2ind = {}
VOCAB_SIZE = [0]

embeddingLen = None

#embeddingMatrix = np.zeros((MAX_WORDS, 100))
#EMBEDDING_FILE = 'embedding'
#embeddingMatrixRef = [ embeddingMatrix ]
#################################################ADD GIT BRANCH
ICAPPF = GITBRANCHPREFIX+'imcap.dat'
embeddingIndex = {}
EMBEDDINGI_FILE = GITBRANCHPREFIX+'embeddingIScaled5'
EMBEDDING_OUT_SCALEFACT = 5 #(-4.0665998, 3.575) needs to be mapped to -1 to +1
embeddingIndexRef = [ embeddingIndex ]


def createDirs():
    try: 
        os.makedirs(GITBRANCHPREFIX)
        os.makedirs(ROOT_DIR + '/' + DIR_IMAGESP)
    except OSError:
        if not os.path.isdir(GITBRANCHPREFIX):
            raise

def badLogs(msg):
    print(msg)
    with open(BADLOGS,"a") as f:
        f.write(msg)

'''
def addToVocab(w):
    global VOCAB_SIZE
    vocab.add(w)
    v_ind2word[ VOCAB_SIZE ] = w
    v_word2ind[ w ] = VOCAB_SIZE
    if not isEmbeddingPresent:
        if w in embeddingIndex.keys():
            embeddingMatrix[VOCAB_SIZE] = embeddingIndex[w]
            print(embeddingMatrix[VOCAB_SIZE])
    if VOCAB_SIZE<10:
        print("%d : %s" % (VOCAB_SIZE, w))
    VOCAB_SIZE += 1
    return VOCAB_SIZE-1
'''
'''
Add NULL Word
Add NonVocab Word
'''  
ENG_SOS = ">"
ENG_EOS = "<"
ENG_EXTRA = "___"
ENG_NONE = "?!?"


'''
def iniVocab():
    global W_SOS,W_EOS
    #addToVocab("none")
    #addToVocab("extra")
    #W_SOS = addToVocab(ENG_SOS)
    #W_EOS = addToVocab(ENG_EOS)
'''


def build_gloveVocab():
    logger.debug("Started")
    if len(embeddingIndexRef[0].keys()) > 0:
        logger.debug("Embedding Already Present %d " % len(embeddingIndexRef[0].keys()))
        return
    isEmbeddingPresent = os.path.exists(EMBEDDINGI_FILE)
    print("Embedding Present %s " % isEmbeddingPresent)
    if isEmbeddingPresent:
        
        '''with open(EMBEDDING_FILE,'r') as f:
            global embeddingMatrix
            embeddingMatrix = pickle.load(f)
            embeddingMatrixRef[0] = embeddingMatrix
        '''
        minVal = float('inf')
        maxVal = -minVal
        with open(EMBEDDINGI_FILE,'r') as f:
            global embeddingIndex
            embeddingIndex = pickle.load(f)
            embeddingIndexRef[0] = embeddingIndex
            for v in embeddingIndex.values():
                for x in v:
                    minVal = min(minVal,x)
                    maxVal = max(maxVal,x)
            #print("minVal, maxVal %s " % str((minVal,maxVal)))
            #exit()
        print("Embedding Loaded")
    else:
        with open(GLOVE_FILE,'r') as f:
            for i,line in enumerate(f):
                tokens = line.split()
                #print(tokens)
                tokens = [tok.__str__() for tok in tokens]
                #print(tokens)
                #exit()
                #if i==200:
                #    break
    
                word = tokens[0]
                #embeddingLen = len(tokens)-1
                if word == "none":
                    print("YoFound you")
                if i<5:
                    print(word)
                    #print(tokens[1:])
                embeddingIndex[word] = np.asarray(tokens[1:], dtype='float32') * (1.0/EMBEDDING_OUT_SCALEFACT)
                #print(embeddingIndex[word])
                #exit()
            #exit()
        assert isEmbeddingPresent == False
        isEmbeddingPresent = True
        #with open(EMBEDDING_FILE,'w') as f:
        #    pickle.dump(embeddingMatrix,f)
        with open(EMBEDDINGI_FILE,'w') as f:
            pickle.dump(embeddingIndex,f)
        print("Embedding Saved!")

    #iniVocab()
    logger.debug("Completed")
            
           
'''
def op_on_caption(cap):
    for w in cap.split(' '):
        w = w.lower()
        if w not in vocab:
            v_word2ind[ w ] = addToVocab(w)
'''
def build_image_caption_pair():
    if os.path.exists(ICAPPF):
        with open(ICAPPF,'r') as f:
            x,mywords = pickle.load(f)
            print("Image Caption Pair Data Model Loaded")
        return x,mywords
    
    x = {}
    logger.debug("Started")
    wordFreq = {}
    uwords = set([])
    with open(FILENAME_CAPTION) as f:
        captions = json.load(f)['annotations']
        count = 0
        for cap in captions:
            cap['caption']= re.sub('[^a-zA-Z]+', ' ', cap['caption'].encode('utf-8')).lower()
            for w in cap['caption'].split(' '):
                if w in uwords:
                    wordFreq[w]+=1
                else:
                    wordFreq[w] =1
                    uwords.add(w)
            #op_on_caption(cap['caption'])
            if True or count < 100:
                x[cap['image_id']] = cap['caption']
            count+=1
    #nmywords = wordFreq.keys()
    #sorted(nmywords, key=lambda key: wordFreq[key], reverse=True)
    #print(wordFreq.keys())
    mywords = set([w for w in wordFreq.keys() if wordFreq[w]>=WORD_MIN_FREQ])
    mywords.add(ENG_SOS)
    mywords.add(ENG_EOS)
    mywords.add(ENG_NONE)
    mywords.add(ENG_EXTRA)

    #print(mywords)
    print(len(mywords))
    #mywords = mywords[:WORD_TOP]
    with open(ICAPPF,'w') as f:
        pickle.dump([x,mywords],f)
        print("Image Caption Pair Data Model Saved")
        
    
    logger.debug("Completed, Vocab Size NONE   ")#%len(v_word2ind))
    return (x,mywords)

#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def imageToVec(_id):
    NEED_W = 224
    NEED_H = 224
    if type("")==type(_id):
        fname = _id
    else:
        fname = get_image_fname(_id)
    #afname = DIR_IMAGESP + fname.split('/')[-1] + '.pickle'
    #if os.path.exists(afname):
    #    with open(afname,'r') as f:
    #        return pickle.load(f)
    #print(fname)
    img = image.load_img(fname, target_size=(NEED_H, NEED_W))
    x = image.img_to_array(img)
    x /= 255.
    x -= 0.5
    x *= 2.
    x = np.asarray(x)
    #with open(afname,'w') as f:
    #    pickle.dump(x,f)
    return x

    ############################################ REMOVE HERE ###
    #img.save("temp.jpg")
    #img = cv2.imread(fname)
    #print(img)
    #img = cv2.resize(img, (NEED_H, NEED_W))
    #cv2.imwrite('test.jpg',img)
    #img = np.asarray(img)
    #print("Shape %s " % (str(np.shape(img))))
    #cv2.imwrite('temp.jpg',img)
    #vec = np.asarray(img)
    #if not vec.any():
    #    badLogs("All zero for %s\n" % str(_id))
    #vec = vec/255.0
    #return vec
    #bw = rgb2gray(img)
    #print("BW Shape %s " % (str(np.shape(bw))))


def getWord2Ind(w):
    w=w.lower()
    if w not in v_word2ind.keys():
        w=ENG_EXTRA
    #print(w)
    return v_word2ind[w]

def word2embd(word):
    if word not in embeddingIndexRef[0].keys():
        word = ENG_EXTRA
    return embeddingIndexRef[0][word]

def embdToWord(embd):
    bestWord = None
    distance = float('inf')
    for word in embeddingIndex.keys():
        e=embeddingIndex[word]
        d = 0
        for a,b in zip(e,embd):
            d+=(a-b)*(a-b)
        if d<distance:
            distance=d
            bestWord = word
    assert(bestWord is not None)
    return (bestWord, distance)

def WordToWordDistance(word1,word2):
    vec1 = word2embd(word1)
    vec2 = word2embd(word2)
    d = 0
    for a,b in zip(vec1,vec2):
        d+=(a-b)*(a-b)
    return d
       

def onehot(vind):
    #print("Vobab Size %d  Ind %d " % (VOCAB_SIZE[0],vind))
    t =  [0]*VOCAB_SIZE[0]
    t[vind] = 1
    return t

def wordToEncode(w, encodeType = None):
    if encodeType is None:
        encodeType = "glove"
    if encodeType == "glove":
        return word2embd(w)
    else:
        return onehot(getWord2Ind(w))

def captionToVec(cap, addOne=False, oneHot=False):
    l = CAPTION_LEN
    if addOne:
        l = l+1
    #print([w.lower() for w in cap.split(' ')])
    cap = cap.lower().split(' ')
    #print(cap)
    cap = cap[:l]
    if len(cap)<l:
        cap.append(ENG_EOS)
    while len(cap)<l:
        cap.append(ENG_NONE)

    if oneHot:
        vec = [wordToEncode(w,"onehot") for w in cap]
    else:
        vec = [wordToEncode(w,"glove") for w in cap]
    return vec


def get_image_caption(_id, lst):
    cap = lst[_id]
    capVec = captionToVec(cap, oneHot=False)
    capVecOneHot = captionToVec(cap, oneHot=True)
    img = imageToVec(_id)
    return np.asarray([img,capVec,capVecOneHot])

def build_vocab():
    lst,topwords = build_image_caption_pair()
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE,'r') as f:
            [v_ind2word_,v_word2ind_, VOCAB_SIZE[0]] = pickle.load(f)
            v_ind2word.clear()
            v_ind2word.update(v_ind2word_)
            v_word2ind.clear()
            v_word2ind.update(v_word2ind_)
            print("Vocab Model Loaded")
    else:
        build_gloveVocab()

        assert len(embeddingIndex.keys())>0
        assert len(v_ind2word) == 0
        assert len(v_word2ind) == 0
        counter = 0
        for w in embeddingIndex.keys():
            if w in topwords:
                v_ind2word[counter]=w
                v_word2ind[w]=counter
                counter += 1
                # ENG_* words present in embeddingIndex and topwords
        VOCAB_SIZE[0] = counter
        #print("Embedding Index Len %d " % len(embeddingIndex.keys()))
        #exit()
        print("TOPWords %d " % len(topwords))
        print("Embeddding Words %d " % len(embeddingIndex.keys()))
        print("Cal Vocab Size %d " % VOCAB_SIZE[0])

        with open(VOCAB_FILE,'w') as f:
            pickle.dump([v_ind2word,v_word2ind, VOCAB_SIZE[0]],f)
            print("Vocab Model Saved")
    
    assert ENG_SOS in v_word2ind.keys()
    assert ENG_EOS in v_word2ind.keys()
    assert ENG_NONE in v_word2ind.keys()
    assert ENG_EXTRA in v_word2ind.keys()
    print("Vocabulary Size %d for %d captions" % (VOCAB_SIZE[0], len(lst)))
    return lst
    
def feed_image_caption(_id,lst):
    img,capGl,capOH = get_image_caption(_id,lst)
    # Glove
    we_sos = [word2embd(ENG_SOS)]
    we_eos = [word2embd(ENG_EOS)]
    # One Hot
    we_eosOH = [wordToEncode(ENG_EOS,encodeType="onehot")]
    return ( (we_sos+list(capGl)), img, (list(capOH) + we_eosOH))
  
def datas_from_ids(idlst,lst):
    images = []
    capS   = []
    capE   = []
    for _id in idlst:
        _capS,_img,_capE = feed_image_caption(_id,lst)
        images.append(_img)
        capS.append(_capS)
        capE.append(_capE)
    return [[np.asarray(capS),np.asarray(images)],np.asarray(capE)]
 
# Train for batch order 0,0,1,0,1,2,0,1,2,3,4,0,1,2,3,4,5..
def data_generator(lst, batch_size, start=0, isTrainSet = True):
    count = (len(lst.keys()))//batch_size
    #print("Max Unique Batches %d " % count)
    countValidation = 5#100
    countTrain = count - 100
    print("Validation Data : %d , Train Batches %d, BatchSize %d\tBatchOffset : %d" % (countValidation, countTrain, batch_size, start))
    offset = 0
    left = countTrain
    extra = 0
    #start = 0
    if not isTrainSet:
        # Validation Data
        left = countValidation
        offset = countTrain * batch_size
        idlst = lst.keys()[offset:offset+left]
        yield datas_from_ids(idlst,lst)
        return
    # Training Data
    maxSequenceLength = countTrain*(countTrain+1)//2
    cbatch = 1
    batchId = 1
    iterBatch = 0

    for it in range(maxSequenceLength):
        if batchId == cbatch:
            batchId = 1  
            cbatch *= 2
            if cbatch > countTrain:
                cbatch = countTrain
        else:
            batchId += 1

        iterBatch+=1
        if iterBatch<=start:
            continue
        idlst = lst.keys()[(batchId-1)*batch_size:(batchId)*batch_size]
        print("Batch Id %d Loaded" % (batchId-1))
        yield datas_from_ids(idlst,lst)
    return

def build_dataset(lst, batch_size = -1, val_size = 0,outerepoch=random.randint(0,10000)):
    logger.debug("Started")

    #_id = lst.keys()[0]
    #imageToVec(_id)
    #capVec =  captionToVec(lst[_id])
    #print(capVec)
    #print("Shape of CapVec %s " % str(np.shape(capVec)))
    train_set = []
    val_set = []
    if batch_size == -1:
        for i,_id in enumerate(lst.keys()):
            if i > 100:
                break
            train_set.append( get_image_caption(_id,lst))
    else:
        tsize = batch_size
        count = (len(lst.keys())-val_size)//tsize
        print("Max Unique Outer Batches %d " % count)
        outerepoch = outerepoch%count
        oinds = outerepoch*tsize
        einds = (outerepoch+1)*tsize
        mylst = lst.keys()[oinds:einds]
        mylst.extend(lst.keys()[-val_size-1:])
        mx = len(mylst)
        #mx = 1000 #########HERE###########
        splitKey =  tsize #int(mx*0.9)

        print("Max Keys %d\tSplit keys %d" % (mx, splitKey))
        todolist = [("Train set",train_set, batch_size,0,splitKey),("Validation Set",val_set, val_size,splitKey,mx-splitKey)]
        for (s,cset, batchsz, offset, datasz) in todolist:
            #indicies = np.random.choice(datasz, batchsz, replace=False)
            #indicies = indicies + offset
            for c,_id in enumerate(mylst[offset:(datasz+offset)]):# enumerate(indicies):
                #_id = lst.keys()[i]
                capimg = get_image_caption(_id,lst)
                #if c==0:
                #    print("%s First Image Id %s with caption : %s " % (s,str(_id), capimg[0]))
                cset.append(capimg)
                if (c*100)%batchsz == 0:
                    print("%s %d %% Loaded!" % (s, c*100/batchsz))
    print("BS %d, VS %d " % (batch_size, val_size))
    print("Shape of Training Set %s " % str(np.shape(train_set)))
    print("Shape of Validation Set %s " % str(np.shape(val_set)))
    logger.debug("Completed")
    return [train_set, val_set]

'''
def train_generator(dataset):
    i = 0
    while i<len(dataset) and i<1:
        out = ( (dataset[i][0], dataset[i][1]), dataset[i][0] )
        print(out)
        yield out#((Caption,Image),Caption)
        i+=1
'''
