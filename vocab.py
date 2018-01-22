from scipy.interpolate import interp1d
from logger import logger
from utils import caption_tokenize
from VideoDataset.videohandler import VideoHandler
import os
import numpy as np
import pickle

# Vocab Size : 1456
class Vocab:
    GLOVE_FILE = 'glove/glove.6B.100d.txt'
    OUTDIM_EMB = 100
    WORD_MIN_FREQ = 5

    def __init__(self,data,data_dir,working_dir,vocab_size = 1456):
        # data = dict(id => captions)
        self.vocab_size = vocab_size
        embeddingI = "%s/glove.dat" % (working_dir)
        glove_file = "%s/%s" % (data_dir, Vocab.GLOVE_FILE)
        self.vocab_file = "%s/vocab.dat" % (working_dir)
        logger.debug("Glove File %s\nEmbedding File %s\nVocab File %s\n" % (glove_file, embeddingI, self.vocab_file))
        self.specialWords = dict()
        self.specialWords['START'] = '>'
        self.specialWords['END'] = '<'
        self.specialWords['NONE'] = '?!?'
        self.specialWords['EXTRA'] = '___'

        self.loadWordEmbedding(embeddingI,glove_file)
        for word,enc in self.specialWords.iteritems():
            assert enc in self.wordEmbedding.keys()
        self.buildVocab(data.values())
        logger.debug("Vocab Build Completed")

    def loadWordEmbedding(self,embeddingI,glove_file):
        isEmbeddingPresent = os.path.exists(embeddingI)
        logger.debug("Embedding Present %s " % isEmbeddingPresent)
        if isEmbeddingPresent:
            with open(embeddingI,'r') as f:
                self.wordEmbedding = pickle.load(f)
            logger.debug("Emdedding Loaded")
        else:
            self.wordEmbedding = dict()
            with open(glove_file,'r') as f:
                for i,line in enumerate(f):
                    tokens = line.split()
                    tokens = [tok.__str__() for tok in tokens]
                    word = tokens[0]
                    self.wordEmbedding[word] = np.asarray(tokens[1:], dtype='float32')
            minVal = float('inf')
            maxVal = -minVal
            for v in self.wordEmbedding.values():
                for x in v:
                    minVal = min(minVal,x)
                    maxVal = max(maxVal,x)
            mapper = interp1d([minVal,maxVal],[-1,1])
            logger.info("Mapping minVal[%f], maxVal[%f] to [-1,1]  " % (minVal,maxVal))
            for w in self.wordEmbedding:
                self.wordEmbedding[w] = mapper(self.wordEmbedding[w])
            print "Cross Check"
            print self.wordEmbedding['good']
            with open(embeddingI,'w') as f:
                pickle.dump(self.wordEmbedding,f)
                logger.info("Embedding Saved!")

    def buildVocab(self,captions):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file,'r') as f:
                self.ind2word = pickle.load(f)
                logger.debug("Vocab Loading from File")
        else:
            x = {}
            for cap in captions:
                for w in caption_tokenize(cap):
                    if w not in x.keys():
                        x[w]=1
                    else:
                        x[w]+=1
            self.ind2word = []
            for w,enc in self.specialWords.iteritems():
                self.ind2word.append(enc)
            self.ind2word.extend([w for w in x.keys() if x[w]>=Vocab.WORD_MIN_FREQ])
            with open(self.vocab_file,'w') as f:
                pickle.dump(self.ind2word,f)
                logger.debug("Vocab File saved")
        logger.info("Vocab Size : %d"%len(self.ind2word))
        self.word2ind = dict()
        for i,w in enumerate(self.ind2word):
            self.word2ind[w]=i
        assert len(self.ind2word) == self.vocab_size

        
def vocabBuilder(datadir, workdir):
    vHandler = VideoHandler(datadir,VideoHandler.fname_train)
    captionData = vHandler.getCaptionData()
    vocab = Vocab(captionData, datadir, workdir)
    return [vHandler, vocab]
    
if __name__ == "__main__":
    vocabBuilder("/home/gagan.cs14/btp","/home/gagan.cs14/btp_VideoCaption")
