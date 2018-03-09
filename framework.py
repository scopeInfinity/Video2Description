import cv2
import shutil, json
from keras import callbacks
import os, sys
import numpy as np
import csv
from vpreprocess import WORKING_DIR
from vpreprocess import  Preprocessor
from logger import logger
from model import VModel
from random import shuffle
from pprint import pformat

CLABEL = 'res_mcnn_rand_b100_s500'
state_uninit = {'epochs':5000, 'start_batch':0, 'batch_size':200, 'saveAtBatch':100, 'steps_per_epoch':500}

MFNAME = WORKING_DIR+'/'+CLABEL+'_model.dat'
_MFNAME = WORKING_DIR+'/'+CLABEL+'_model.dat.bak'
STATE = WORKING_DIR+'/'+CLABEL+'_state.txt'
RESULTS = WORKING_DIR+'/'+CLABEL+'_results.txt'
FRESTART = WORKING_DIR+'/restart'

class TrainingLogs:
    def __init__(self, prefix=""):
        self.epochLogHistory = []
        self.fname = WORKING_DIR+'/'+CLABEL + "_logs_" + prefix + ".txt"

    def flush(self):
        if not os.path.exists(self.fname):
            with open(self.fname, "w") as f:
                wr = csv.writer(f)
        if len(self.epochLogHistory) > 0:
            with open(self.fname, "a") as f:
                wr = csv.writer(f)
                for h in self.epochLogHistory:
                    wr.writerow(h)
            self.epochLogHistory = []
            logger.debug("Training Logs flushed")

    def add(self,cont):
        MXCol = 15
        dat = [-1] * 15
        for i in range(min(MXCol,len(cont))):
            dat[i]=cont[i]
        self.epochLogHistory.append(dat)

class ModelGeneratorCallback(callbacks.Callback):

    def __init__(self, state, tlogs, elogs, framework):
        self.state = state
        self.lastloss = str('inf')
        self.tlogs = tlogs
        self.elogs = elogs
        self.framework = framework
        self.batchTrainedCounter = 0

    def on_epoch_end(self, epoch, logs={}):
        logger.debug("Epoch %d End " % epoch)
        self.state['epochs']-=1
        loss = logs['loss']
        acc  = logs['acc']
        valloss = logs['val_loss']
        valacc  = logs['val_acc']
        self.elogs.add([epoch,loss, acc, valloss, valacc])
        self.elogs.flush()
        self.framework.save(epoch=("%03d_loss_%s" % (self.state['epochs'],str(valloss))))
        return

    def on_batch_end(self, batch, logs={}):
        logger.debug("Batch %d ends" % batch)
        valloss = -1
        valacc  = -1
        loss = logs['loss']
        acc  = logs['acc']
        self.lastloss = loss
        print "Keys Logger %s " % str(logs.keys())
        self.tlogs.add([batch, loss, acc, valloss, valacc])
        self.state['start_batch'] += 1
        self.batchTrainedCounter += 1
        logger.debug("Batches Trained : %d" % self.batchTrainedCounter)
        if self.batchTrainedCounter % self.state['saveAtBatch'] == 0:
            logger.debug("Preparing To Save")
            self.framework.save()
            self.tlogs.flush()
        
class Framework():
    
    def __init__(self):
        self.state = state_uninit
        self.tlogs = TrainingLogs()
        self.elogs = TrainingLogs(prefix = "epoch_")
        self.model = None          # Init in self.build_model()
        self.preprocess = Preprocessor()
        self.build_model()
        self.load()
        logger.debug("__init__ framework complete")

    def build_model(self):
        vocab = self.preprocess.vocab
        self.vmodel = VModel(vocab.CAPTION_LEN, vocab.VOCAB_SIZE)
        self.model = self.vmodel.get_model()
        assert self.preprocess is not None
        self.preprocess.set_vmodel(self.vmodel)

    def load(self):
        if os.path.exists(MFNAME):
            self.model.load_weights(MFNAME)
            logger.debug("Weights Loaded")
        if os.path.exists(STATE):
            with open(STATE) as f:
                self.state = json.load(f)
                logger.debug("State Loaded")

    def save(self, epoch='xx'):
        try:
            pass
        finally:
            tname = _MFNAME
            self.model.save_weights(tname)
            fname = MFNAME
            if epoch != 'xx':
                fname = MFNAME + '_' + epoch
            shutil.copy2(tname,fname)
            os.remove(tname)
            logger.debug("Weights Saved")
            with open(STATE,'w') as f:
                json.dump(self.state,f)
                logger.debug("State Saved")

    def train_generator(self):
        epochs = self.state['epochs']
        bs = self.state['batch_size']
        steps_per_epoch = self.state['steps_per_epoch']
        validation_steps = 1
        logger.debug("Epochs Left : %d " % epochs)
        logger.debug("Batch Size  : %d " % bs)

        train_dg = self.preprocess.data_generator(bs, start=self.state['start_batch'], typeSet = 0)
        val_dg = self.preprocess.data_generator(bs, -1, typeSet = 1)
        logger.debug("Attemping to fit")
        callbacklist = [ModelGeneratorCallback(self.state, self.tlogs, self.elogs, self)]
        self.model.fit_generator(train_dg, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,validation_data=val_dg, validation_steps=validation_steps, initial_epoch=0, callbacks=callbacklist)

    def predict_model_direct(self, fnames, cache_ids = None):
        videoVecs = []
        for i in range(len(fnames)):
            cid = None
            if cache_ids is not None:
                cid = cache_ids[i]
            videoVecs.append(self.preprocess.get_video_content(fnames[i], cache_id = cid))
        videoVecs =np.array(videoVecs)
        # videoVecs =np.array([self.preprocess.get_video_content(f) for f  in fnames])
        count = len(fnames)
        for i,video in enumerate(videoVecs):
            if video is None:
                return None,{'error':'Video %d couldn\'t be loaded. %s ' % (i, fnames[i])}
        logger.debug("Predicting for Videos :- \n\t%s " % fnames)
        l = 0
        vocab = self.preprocess.vocab
        startCapRow = [vocab.wordEmbedding[vocab.specialWords['START']] ]
        startCapRow.extend([ vocab.wordEmbedding[vocab.specialWords['NONE']] ] * vocab.CAPTION_LEN)
        
        embeddedCap = np.array([ startCapRow  ] * count)
        logger.debug("Shape of Caption : %s", str(np.shape(embeddedCap)))
        stringCaption = []
        for i in range(count):
            stringCaption.append([])
        while l < vocab.CAPTION_LEN:
            newOneHotCap = self.model.predict([embeddedCap, videoVecs])
            print "Shape of out Predict Model : %s " % str(np.shape(newOneHotCap))
            for i,newOneHotWord in enumerate(newOneHotCap):
                nword = vocab.word_fromonehot(newOneHotWord[l])
                # print str(i)+" "+str(l)+" "+nword
                stringCaption[i].append( nword )
                if l + 1 != vocab.CAPTION_LEN:
                    embeddedCap[i][l+1] = vocab.wordEmbedding[nword]

            print [' '.join(cap) for cap in stringCaption]
            l += 1
        logger.debug("Prediction Complete")
        captionObject = []
        for i,cap in enumerate(stringCaption):
            captionObject.append({'fname':fnames[i], 'caption':cap})
        return stringCaption, captionObject

    def predict_ids(self, _ids):
        result = self.predict_model(_ids = _ids)
        return result

    def predict_fnames(self, fnames):
        result = self.predict_model(fnames = fnames)
        return result

    def predict_model(self, _ids = None, fnames = None):
        assert (_ids is None) ^ (fnames is None)
        vHandler = self.preprocess.vHandler
        if fnames is None:
            fnames = []
            for _id in _ids:
                logger.debug("Obtaining fname for %d" % _id)
                fname = vHandler.downloadVideo(_id)
                if fname is None:
                    logger.info("Ignoring %d video " % _id)
                else:
                    fnames.append(fname)
        predictions,output = self.predict_model_direct(fnames, cache_ids = _ids)
        results = []
        for i in range(len(fnames)):
            print()
            logger.debug("For eog %s" % fnames[i])
            predictedCaption = ' '.join(predictions[i])
            logger.debug("Predicted Caption : %s" % predictedCaption )
            actualCaption = None
            if _ids is not None:
                actualCaption = vHandler.getCaptionData()[_ids[i]]
                logger.debug("Actual Captions - \n%s" % pformat(actualCaption) )
            res = dict()
            res['fname'] = fnames[i]
            res['output'] = predictedCaption
            res['actual'] = actualCaption
            results.append(res)
        return json.dumps(results, indent=4, sort_keys=True)
                                        
    def isVideoExtension(self, fname):
        for ext in ['mp4','jpeg','png']:
            if fname.endswith('.'+ext):
                return True
        return False

    def predict_test(self, dirpath, mxc):
        videos = ["%s/%s" % (dirpath,vid) for vid in os.listdir(dirpath) if self.isVideoExtension(vid)][0:mxc]
        self.predict_model(fnames = videos)

    def clean_caption(self, msg):
        if '<' in msg:
            return msg.split("<")[0]
        return msg

    def save_all(self, _ids):
        _result = json.loads(self.predict_ids(_ids))
        test_predicted = []
        test_actual = []
        for res in _result:
            tp = dict()
            _id = int(res['fname'].split('/')[-1].split('.')[0])
            tp['video_id'] =  _id
            tp['caption'] =  self.clean_caption(res['output'])
            test_predicted.append(tp)

            for cap in res['actual']:
                tp_actual = dict()
                tp_actual['video_id'] = _id
                tp_actual['caption'] = cap
                test_actual.append(tp_actual)
        result = dict()
        result['predicted'] = test_predicted
        result['actual'] = test_actual

        with open(RESULTS, 'w') as f:
            f.write(json.dumps(result))
        logger.debug("Result Saved")

    def get_testids(self, count = -1):
        ids = self.preprocess.vHandler.getTestIds()
        if count == -1:
            count = len(ids)
        else:
            shuffle(ids)
        return ids[:count]

    def get_trainids(self, count = -1):
        ids = self.preprocess.vHandler.getTrainingIds()
        if count == -1:
            count = len(ids)
        else:
            shuffle(ids)
        return ids[:count]

'''
#deprecated
def main():
    framework = Framework()                                            
    if len(sys.argv) == 2 and '-train' == sys.argv[1]:
        framework.train_generator()
    elif len(sys.argv) == 3 and '-p_ids' == sys.argv[1]:
        framework.predict_model(_ids = [int(x) for x in sys.argv[2].split(",")])
    elif len(sys.argv) == 3 and '-prand_test' == sys.argv[1]:
        framework.predict_model(_ids = framework.get_testids(int(sys.argv[2])))
    elif len(sys.argv) == 3 and '-prand_train' == sys.argv[1]:
        framework.predict_model(_ids = framework.get_trainids(int(sys.argv[2])))
    elif len(sys.argv) == 3 and '-p_names' == sys.argv[1]:
        framework.predict_model(fnames = sys.argv[2].split(","))
    elif len(sys.argv) == 4 and '-ptest' == sys.argv[1]:
        framework.predict_test(sys.argv[2],int(sys.argv[3]))
    else:
        print "Invalid Argument"

if __name__ == '__main__':
    main()
'''
