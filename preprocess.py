import pickle
import sys
from sets import Set
import json,re 
import os, shutil
import cv2
from keras.preprocessing import image
#from PIL import Image as pil_image
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
'''
vocab = Set([])
v_ind2word = {}
v_word2ind = {}
VOCAB_SIZE = 0
embeddingLen = None
#embeddingMatrix = np.zeros((MAX_WORDS, 100))
#EMBEDDING_FILE = 'embedding'
#embeddingMatrixRef = [ embeddingMatrix ]
'''
ICAPPF = 'imcap.dat'
embeddingIndex = {}
EMBEDDINGI_FILE = 'embeddingIScaled5'
EMBEDDING_OUT_SCALEFACT = 5 #(-4.0665998, 3.575) needs to be mapped to -1 to +1
isEmbeddingPresent = os.path.exists(EMBEDDINGI_FILE)
embeddingIndexRef = [ embeddingIndex ]

print "Embedding Present %s " % isEmbeddingPresent

def badLogs(msg):
    print msg
    with open("badlogs.txt","a") as f:
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
            print embeddingMatrix[VOCAB_SIZE]
    if VOCAB_SIZE<10:
        print "%d : %s" % (VOCAB_SIZE, w)
    VOCAB_SIZE += 1
    return VOCAB_SIZE-1
'''
'''
Add NULL Word
Add NonVocab Word
'''  
ENG_SOS = "start"
ENG_EOS = "end"
ENG_NONE = "none"
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
            #print "minVal, maxVal %s " % str((minVal,maxVal))
            #exit()
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
                #embeddingLen = len(tokens)-1
                if word == "none":
                    print "YoFound you"
                if i<5:
                    print word
                    #print tokens[1:]
                embeddingIndex[word] = np.asarray(tokens[1:], dtype='float32') * (1.0/EMBEDDING_OUT_SCALEFACT)
                #print embeddingIndex[word]
                #exit()
            #exit()
        global isEmbeddingPresent
        assert isEmbeddingPresent == False
        isEmbeddingPresent = True
        #with open(EMBEDDING_FILE,'w') as f:
        #    pickle.dump(embeddingMatrix,f)
        with open(EMBEDDINGI_FILE,'w') as f:
            pickle.dump(embeddingIndex,f)
        print "Embedding Saved!"

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
            x = pickle.load(f)
            print "Image Caption Pair Data Model Loaded"
        return x
    
    x = {}
    logger.debug("Started")
    with open(FILENAME_CAPTION) as f:
        captions = json.load(f)['annotations']
        count = 0
        for cap in captions:
            cap['caption']= re.sub('[^a-zA-Z]+', ' ', cap['caption'].encode('utf-8'))

            #op_on_caption(cap['caption'])
            if True or count < 100:
                x[cap['image_id']] = cap['caption']
            count+=1
    with open(ICAPPF,'w') as f:
        pickle.dump(x,f)
        print "Image Caption Pair Data Model Saved"
        
    
    logger.debug("Completed, Vocab Size NONE ")
    return x

#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def imageToVec(_id):
    NEED_W = 224
    NEED_H = 224
    fname = get_image_fname(_id)
    print fname
    img = image.load_img(fname, target_size=(NEED_H, NEED_W))
    ############################################ REMOVE HERE ###
    #img.save("temp.jpg")
    #img = cv2.imread(fname)
    #print img
    #img = cv2.resize(img, (NEED_H, NEED_W))
    #cv2.imwrite('test.jpg',img)
    #img = np.asarray(img)
    #print "Shape %s " % (str(np.shape(img)))
    #cv2.imwrite('temp.jpg',img)
    vec = np.asarray(img)
    if not vec.any():
        badLogs("All zero for %s\n" % str(_id))
    vec = vec/255.0
    return vec
    #bw = rgb2gray(img)
    #print "BW Shape %s " % (str(np.shape(bw)))


#def getWord2Ind(w):
#    w=w.lower()
#    if w not in v_word2ind.keys():
#        w='extra'
#    return v_word2ind[w]

def word2embd(word):
    if word not in embeddingIndexRef[0].keys():
        word = ENG_NONE
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
       
def captionToVec(cap, addOne=False):
    l = CAPTION_LEN
    if addOne:
        l = l+1
    #print [w.lower() for w in cap.split(' ')]
    cap = cap.lower().split(' ')
    print cap
    cap = cap[:l]
    while len(cap)<l:
        cap.append(ENG_EOS)
    vec = [word2embd(w) for w in cap]
#    nums = [getWord2Ind(w) for w in cap.split(' ')]
#    vec = sequence.pad_sequences([nums],maxlen=CAPTION_LEN,padding='post', truncating='post')[0]
    
    return vec

'''
def onehot(vec):
    vector = []
    for i in vec:
        t =  [0]*VOCAB_SIZE
        t[i] = 1
        vector.append( t )
    return np.asarray(vector)
'''

def get_image_caption(_id, lst):
    cap = lst[_id]
    capVec = captionToVec(cap)
    img = imageToVec(_id)
    return np.asarray([capVec,img])

def build_vocab():
    lst = build_image_caption_pair()
    print "Vocabulary Size NONE for %d captions" % (len(lst))
    return lst
    
def build_dataset(lst, batch_size = -1, val_size = 0):
    logger.debug("Started")

    #_id = lst.keys()[0]
    #imageToVec(_id)
    #capVec =  captionToVec(lst[_id])
    #print capVec
    #print "Shape of CapVec %s " % str(np.shape(capVec))
    train_set = []
    val_set = []
    if batch_size == -1:
        for i,_id in enumerate(lst.keys()):
            if i > 100:
                break
            train_set.append( get_image_caption(_id,lst))
    else:
        mx = len(lst.keys())
        mx = 1000 #########HERE###########
        splitKey = int(mx*0.9)
        print "Max Keys %d\tSplit keys %d" % (mx, splitKey)
        todolist = [("Train set",train_set, batch_size,0,splitKey),("Validation Set",val_set, val_size,splitKey,mx-splitKey)]
        for (s,cset, batchsz, offset, datasz) in todolist:
            indicies = np.random.choice(datasz, batchsz, replace=False)
            indicies = indicies + offset
            for c,i in enumerate(indicies):
                _id = lst.keys()[i]
                capimg = get_image_caption(_id,lst)
                if c==0:
                    print "%s First Image Id %s with caption : %s " % (s,str(_id), capimg[0])
                cset.append(capimg)
    print "BS %d, VS %d " % (batch_size, val_size)
    print "Shape of Training Set %s " % str(np.shape(train_set))
    print "Shape of Validation Set %s " % str(np.shape(val_set))
    logger.debug("Completed")
    return [train_set, val_set]

'''
def train_generator(dataset):
    i = 0
    while i<len(dataset) and i<1:
        out = ( (dataset[i][0], dataset[i][1]), dataset[i][0] )
        print out
        yield out#((Caption,Image),Caption)
        i+=1
'''
