import pickle
import sys
from sets import Set
import json,re 
import os, shutil
import cv2
import numpy as np
from logger import logger
from keras.preprocessing import sequence
from keras import callbacks

# Parameters
CAPTION_LEN = 10
MAX_WORDS = 400000

os.chdir('/home/gagan.cs14/btp')
FILENAME_CAPTION = 'ImageDataset/annotations/captions_train2014.json'
DIR_IMAGES = 'ImageDataset/train2014/'
GLOVE_FILE = 'glove/glove.6B.100d.txt'
OUTDIM_EMB = 100
USE_GLOVE = True
def get_image_fname(_id):
    return '%sCOCO_train2014_%012d.jpg' % (DIR_IMAGES, _id)

vocab = Set([])
v_ind2word = {}
v_word2ind = {}
VOCAB_SIZE = 0
embeddingIndex = {}
embeddingLen = None
embeddingMatrix = np.zeros((MAX_WORDS, 100))
EMBEDDING_FILE = 'embedding'
EMBEDDINGI_FILE = 'embeddingI'
isEmbeddingPresent = os.path.exists(EMBEDDING_FILE)
embeddingMatrixRef = [ embeddingMatrix ]
embeddingIndexRef = [ embeddingIndex ]

#print isEmbeddingPresent

def addToVocab(w):
    global VOCAB_SIZE
    vocab.add(w)
    v_ind2word[ VOCAB_SIZE ] = w
    v_word2ind[ w ] = VOCAB_SIZE
    if not isEmbeddingPresent:
        if w in embeddingIndex.keys():
            embeddingMatrix[VOCAB_SIZE] = embeddingIndex[w]
            #print embeddingMatrix[VOCAB_SIZE]
    if VOCAB_SIZE<10:
        print "%d : %s" % (VOCAB_SIZE, w)
    VOCAB_SIZE += 1
    return VOCAB_SIZE-1
'''
Add NULL Word
Add NonVocab Word
'''  
ENG_SOS = "start"
ENG_EOS = "end"
def iniVocab():
    global W_SOS,W_EOS
    #addToVocab("none")
    #addToVocab("extra")
    #W_SOS = addToVocab(ENG_SOS)
    #W_EOS = addToVocab(ENG_EOS)



def build_gloveVocab():
    logger.debug("Started")
    if isEmbeddingPresent:
        with open(EMBEDDING_FILE,'r') as f:
            global embeddingMatrix
            embeddingMatrix = pickle.load(f)
            embeddingMatrixRef[0] = embeddingMatrix
        with open(EMBEDDINGI_FILE,'r') as f:
            global embeddingIndex
            embeddingIndex = pickle.load(f)
            embeddingIndexRef[0] = embeddingIndex
        print "Embedding Loaded"
    else:
        with open(GLOVE_FILE,'r') as f:
            for i,line in enumerate(f):
                tokens = line.split()
                #print tokens
                tokens = [tok.__str__() for tok in tokens]
                #print tokens
                #exit()
                #if i==200:
                #    break
    
                word = tokens[0]
                embeddingLen = len(tokens)-1
                if word == "none":
                    print "YoFound you"
                if i<5:
                    print word
                    #print tokens[1:]
                embeddingIndex[word] = np.asarray(tokens[1:], dtype='float32')
            #exit()
    iniVocab()
    logger.debug("Completed")
            
build_gloveVocab() 
           
def op_on_caption(cap):
    for w in cap.split(' '):
        w = w.lower()
        if w not in vocab:
            v_word2ind[ w ] = addToVocab(w)

def build_image_caption_pair():
    x = {}
    logger.debug("Started")
    with open(FILENAME_CAPTION) as f:
        captions = json.load(f)['annotations']
        count = 0
        for cap in captions:
            cap['caption']= re.sub('[^a-zA-Z]+', ' ', cap['caption'].encode('utf-8'))

            op_on_caption(cap['caption'])
            if True or count < 100:
                x[cap['image_id']] = cap['caption']
            count+=1
    global isEmbeddingPresent
    if not isEmbeddingPresent:
        isEmbeddingPresent = True
        with open(EMBEDDING_FILE,'w') as f:
            pickle.dump(embeddingMatrix,f)
        with open(EMBEDDINGI_FILE,'w') as f:
            pickle.dump(embeddingIndex,f)
        print "Embedding Saved!"

    logger.debug("Completed, Vocab Size %s " % VOCAB_SIZE)
    return x

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def imageToVec(_id):
    NEED_W = 224
    NEED_H = 224
    fname = get_image_fname(_id)
    print fname
    img = cv2.imread(fname)
    #print img
    img = cv2.resize(img, (NEED_H, NEED_W))
    cv2.imwrite('test.jpg',img)
    img = np.asarray(img)
    #print "Shape %s " % (str(np.shape(img)))
    return img
    bw = rgb2gray(img)
    print "BW Shape %s " % (str(np.shape(bw)))


def getWord2Ind(w):
    w=w.lower()
    if w not in v_word2ind.keys():
        w='extra'
    return v_word2ind[w]
def captionToVec(cap):
    #print [w.lower() for w in cap.split(' ')]
    print cap
    nums = [getWord2Ind(w) for w in cap.split(' ')]
    vec = sequence.pad_sequences([nums],maxlen=CAPTION_LEN,padding='post', truncating='post')[0]
    return vec

def onehot(vec):
    vector = []
    for i in vec:
        t =  [0]*VOCAB_SIZE
        t[i] = 1
        vector.append( t )
    return np.asarray(vector)

def word2embd(word):
    if word not in embeddingIndexRef[0].keys():
        word = "none"
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
    return bestWord

def get_image_caption(_id, lst):
    cap = lst[_id]
    capVec = captionToVec(cap)
    img = imageToVec(_id)
    return np.asarray([capVec,img])

def build_vocab():
    global lst
    lst = build_image_caption_pair()
    print "Vocabulary Size %d for %d captions" % (len(vocab),len(lst))
    
def build_dataset(batch_size = -1):
    logger.debug("Started")

    #_id = lst.keys()[0]
    #imageToVec(_id)
    #capVec =  captionToVec(lst[_id])
    #print capVec
    #print "Shape of CapVec %s " % str(np.shape(capVec))
    train_set = []
    test_set = []
    if batch_size == -1:
        for i,_id in enumerate(lst.keys()):
            if i > 100:
                break
            train_set.append( get_image_caption(_id,lst))
    else:
        mx = len(lst.keys())
        mx = 1000 #########HERE###########
        indicies = np.random.choice(mx, batch_size, replace=False)
        for c,i in enumerate(indicies):
            _id = lst.keys()[i]
            capimg = get_image_caption(_id,lst)
            if c==0:
                print "First Image Id %s with caption : %s " % (str(_id), capimg[0])
            train_set.append( capimg )

    print "Shape of Training Set %s " % str(np.shape(train_set))
    logger.debug("Completed")
    return [train_set, test_set]

def train_generator(dataset):
    i = 0
    while i<len(dataset) and i<1:
        out = ( (dataset[i][0], dataset[i][1]), dataset[i][0] )
        print out
        yield out#((Caption,Image),Caption)
        i+=1
