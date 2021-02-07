import ast
import csv
import json
import numpy as np
import os
import shutil
import sys

from keras import callbacks
from pprint import pformat
from random import shuffle

from backend.model import VModel
from backend.vpreprocess import  Preprocessor
from common.config import get_app_config, get_vpreprocess_config
from common.logger import logger
from common.status import ModelWeightsStatus


WORKERS = 40
DATASET_CACHE = get_app_config()["DATASET_CACHE"]
COCOFNAME = get_vpreprocess_config()["COCOFNAME"]

CLABEL = 'ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS'

state_uninit = {'epochs':5000, 'start_batch':0, 'batch_size':100, 'saveAtBatch':500, 'steps_per_epoch':500}

MFNAME = DATASET_CACHE+'/'+CLABEL+'_model.dat'
_MFNAME = DATASET_CACHE+'/'+CLABEL+'_model.dat.bak'
STATE = DATASET_CACHE+'/'+CLABEL+'_state.txt'
RESULTS = DATASET_CACHE+'/'+CLABEL+'_results.txt'
FRESTART = DATASET_CACHE+'/restart'
PREDICT_BATCHSIZE = 200

class TrainingLogs:
    def __init__(self, prefix=""):
        self.epochLogHistory = []
        self.fname = DATASET_CACHE+'/'+CLABEL + "_logs_" + prefix + ".txt"

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
        self.lastloss = float('inf')
        self.tlogs = tlogs
        self.elogs = elogs
        self.last_epochmodel = None
        self.framework = framework
        self.batchTrainedCounter = 0
        self.bestlossepoch = float('inf')

    def on_epoch_end(self, epoch, logs={}):
        logger.debug("Epoch %d End " % epoch)
        self.state['epochs']-=1
        loss = logs['loss']
        acc  = logs['acc']
        valloss = logs['val_loss']
        valacc  = logs['val_acc']
        # Sample Content
        # {'CIDEr': 0.11325126353463148, 'Bleu_4': 0.1706107390467726, 'Bleu_3': 0.27462591349020055, 'Bleu_2': 0.4157995334621001, 'Bleu_1': 0.6064295446876932, 'ROUGE_L': 0.40471970665189977, 'METEOR': 0.17162570735633326}
        coco_json = self.framework.eval_onvalidation()
        cider = coco_json['CIDEr']
        bleu4 = coco_json['Bleu_4']
        rouge = coco_json['ROUGE_L']
        meteor = coco_json['METEOR']
        ename = "%.3f_Cider%.3f_Blue%.3f_Rouge%.3f_Meteor%.3f" % (valloss, cider, bleu4, rouge, meteor)
        self.elogs.add([epoch,loss, acc, valloss, valacc, cider, bleu4, rouge, meteor])
        self.elogs.flush()
        if valloss < self.bestlossepoch or True:
            to_rm = self.last_epochmodel
            self.last_epochmodel = self.framework.save(epoch=("%03d_loss_%s" % (self.state['epochs'],ename)))
            self.bestlossepoch = valloss
            if to_rm is not None:
                pass
                # os.remove(to_rm)
        return

    def on_batch_end(self, batch, logs={}):
        logger.debug("Batch %d ends" % batch)
        valloss = -1
        valacc  = -1
        loss = logs['loss']
        acc  = logs['acc']
        self.lastloss = loss
        print("Keys Logger %s " % str(logs.keys()))
        self.tlogs.add([batch, loss, acc, valloss, valacc])
        self.state['start_batch'] += 1
        self.batchTrainedCounter += 1
        logger.debug("Batches Trained : %d" % self.batchTrainedCounter)
        if self.batchTrainedCounter % self.state['saveAtBatch'] == 0:
            logger.debug("Preparing To Save")
            self.framework.save()
            self.tlogs.flush()
        

class Framework():
    
    def __init__(self, model_load = MFNAME, train_mode = False):
        self.mode_learning = train_mode
        self.state = state_uninit
        self.file_model = model_load
        self.status_model_weights = ModelWeightsStatus.NO_INFO
        self.tlogs = TrainingLogs()
        self.elogs = TrainingLogs(prefix = "epoch_")
        self.model = None          # Init in self.build_model()
        self.preprocess = Preprocessor()
        self.build_model()
        self.load()
        logger.debug("__init__ framework complete")

    def build_model(self):
        vocab = self.preprocess.vocab
        self.vmodel = VModel(vocab.CAPTION_LEN, vocab.VOCAB_SIZE, learning = self.mode_learning)
        self.model = self.vmodel.get_model()
        assert self.preprocess is not None
        self.preprocess.set_vmodel(self.vmodel)

    def load(self):
        logger.debug("Model Path: %s" % self.file_model)
        if os.path.exists(self.file_model):
            self.model.load_weights(self.file_model)
            self.status_model_weights = ModelWeightsStatus.SUCCESS
            logger.debug("Weights Loaded")
        else:
            self.status_model_weights = ModelWeightsStatus.MODEL_NOT_FOUND
            logger.warning("Weights files not found.")
        if os.path.exists(STATE):
            with open(STATE) as f:
                self.state = json.load(f)
                logger.debug("State Loaded")

    def get_weights_status(self):
        return str(self.status_model_weights)

    def save(self, epoch='xx'):
        try:
            pass
        finally:
            tname = _MFNAME
            self.model.save_weights(tname)
            fname = self.file_model
            if epoch != 'xx':
                fname = self.file_model + '_' + epoch
            shutil.copy2(tname,fname)
            os.remove(tname)
            logger.debug("Weights Saved")
            with open(STATE,'w') as f:
                json.dump(self.state,f)
                logger.debug("State Saved")
            return fname
        return None

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
        self.vmodel.train_mode()
        self.model.fit_generator(train_dg, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 verbose=1,validation_data=val_dg, validation_steps=validation_steps,
                                 initial_epoch=0, callbacks=callbacklist,
                                 workers=WORKERS, use_multiprocessing=True)

    def predict_model_direct(self, fnames, cache_ids = None):
        videoVecs = []
        audioVecs = []
        for i in range(len(fnames)):
            cid = None
            if cache_ids is not None:
                cid = cache_ids[i]
            vid_audio = self.preprocess.get_video_content(fnames[i], cache_id = cid)
            if vid_audio is None:
                return None,{'error':'Video %d couldn\'t be loaded. %s ' % (i, fnames[i])}
            videoVecs.append(vid_audio[0]) # Video Features
            audioVecs.append(vid_audio[1]) # Audio Features
        videoVecs = np.array(videoVecs)
        audioVecs = np.array(audioVecs)

        # videoVecs =np.array([self.preprocess.get_video_content(f) for f  in fnames])
        count = len(fnames)
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
            newOneHotCap = self.model.predict([embeddedCap, audioVecs, videoVecs])
            print("Shape of out Predict Model : %s " % str(np.shape(newOneHotCap)))
            for i,newOneHotWord in enumerate(newOneHotCap):
                nword = vocab.word_fromonehot(newOneHotWord[l])
                # print(str(i)+" "+str(l)+" "+nword)
                stringCaption[i].append( nword )
                if l + 1 != vocab.CAPTION_LEN:
                    embeddedCap[i][l+1] = vocab.wordEmbedding[nword]

            print([' '.join(cap) for cap in stringCaption])
            l += 1
        logger.debug("Prediction Complete")
        captionObject = []
        for i,cap in enumerate(stringCaption):
            captionObject.append({'fname':fnames[i], 'caption':cap})
        return stringCaption, captionObject

    def predict_ids(self, _ids):
        logger.debug("Trying to predict for %s" % (_ids,))
        result = self.predict_model(_ids = _ids)
        return result

    def predict_fnames(self, fnames):
        logger.debug("Trying to predict for %s" % (fnames,))
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

        batch_size = PREDICT_BATCHSIZE
        batch_count = (len(fnames)+batch_size-1)//batch_size
        predictions,output = ([],[])
        for i in range(batch_count):
            cids = None
            if _ids is not None:
                cids = _ids[i*batch_size:(i+1)*batch_size]
            pred,out = self.predict_model_direct(fnames[i*batch_size:(i+1)*batch_size], cache_ids = cids)
            if pred is None:
                logger.debug(json.dumps(out))
                assert False
            predictions.extend(pred)
            output.extend(out)
        results = []
        for i in range(len(fnames)):
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

    def save_all(self, _ids, save = RESULTS):
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
        with open(save, 'w') as f:
            f.write(json.dumps(result))
        logger.debug("Result Saved")
    
    def eval_onvalidation(self):
        fname = '/tmp/save_model_' + CLABEL
        logger.debug("Calculating cocoscore")
        valids = self.preprocess.vHandler.getValidationIds()
        self.save_all(valids, save = fname)
        cmd = "python %s %s | tail -n 1" % (COCOFNAME, fname)
        coco = ast.literal_eval(os.popen(cmd).read().strip())
        logger.debug("Done")
        logger.debug("Coco Scores :%s\n" % json.dumps(coco,indent=4, sort_keys=True))
        return coco

    def get_testids(self, count = -1):
        ids = self.preprocess.vHandler.getTestIds()
        if count == -1:
            count = len(ids)
        else:
            shuffle(ids)
        return ids[:count]

    def get_valids(self, count = -1):
        ids = self.preprocess.vHandler.getValidationIds()
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
