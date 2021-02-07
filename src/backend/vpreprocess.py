import numpy as np
import random
import os

from keras.preprocessing import image

from common.config import get_vpreprocess_config
from common.logger import logger
from backend.vocab import vocabBuilder


BADLOGS = get_vpreprocess_config()["LOGS_DIR"]

def badLogs(msg):
    logger.debug(msg)
    with open(BADLOGS,"a") as f:
        f.write(msg)

class Preprocessor:
    def __init__(self):
        self.vHandler,self.vocab = vocabBuilder()

    def set_vmodel(self, vmodel):
        self.vHandler.set_vmodel(vmodel)

    def imageToVec(self, fname):
        NEED_W = 224
        NEED_H = 224
        img = image.load_img(fname, target_size=(NEED_H, NEED_W))
        x = image.img_to_array(img)
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    '''
    Either convert videos from ids or frame file names
    '''
    COUNTER = 0
    def videoToVec(self, _id = None, vfname = None, cache_id = None):
        assert (_id is None) ^ (vfname is None)
        if not _id == None:
            out = self.vHandler.get_iframes_audio(_id = _id, logs = False)
        else:
            out = self.vHandler.get_iframes_audio(sfname = vfname, logs = False, cache_id = cache_id)
        if out is None:
            return None
        (frames, afeatures) = out
        return frames, afeatures
        # deprecated
        fnames = None
        edir = None
        if fnames is None:
            ef = self.vHandler.get_frames(_id = _id, logs = False)
            if ef is not None:
                edir, fnames = ef
        if fnames is None:
            return None
        content = []
        for i,fname in enumerate(fnames):
            content.append(self.imageToVec(fname))
        self.vHandler.free_frames(edir)

        #if len(fnames)>0:
        #    os.system("cp \"%s\" ~/TESTING/%04d.jpg" % (fnames[0],Preprocessor.COUNTER))
        #    Preprocessor.COUNTER += 1
        return content

    def get_video_content(self, vfname, cache_id = None):
        return self.videoToVec(vfname = vfname, cache_id = cache_id)

    def get_video_caption(self, _id, just_one_caption = True):
        vid_a = self.videoToVec(_id = _id)
        if vid_a is None:
            return None
        (vid, afeature) = vid_a
        data = self.vHandler.getCaptionData()
        out = []
        for cur_caption in data[_id]:
            captionIn = self.vocab.get_caption_encoded(cur_caption, True, True, False)
            captionOut = self.vocab.get_caption_encoded(cur_caption, False, False, True)
            out.append([afeature,vid,captionIn,captionOut])
        if len(out) == 0:
            return None
        if just_one_caption:
            return [random.choice(out)]
        return out

    def datas_from_ids(self, idlst):
        logger.debug("\n Loading Video/Captions for ids : %s" % str(idlst))
        afeatures = []
        vids   = []
        capIn  = []
        capOut = []
        for _id in idlst:
            vccs = self.get_video_caption(_id, just_one_caption = True)
            if vccs is None:
                continue
            for vcc in vccs:
                _afeature, _vid, _capIn, _capOut = vcc
                afeatures.append(_afeature)
                vids.append(_vid)
                capIn.append(_capIn)
                capOut.append(_capOut)
        afeatures  = np.asarray(afeatures)
        capIn  = np.asarray(capIn)
        capOut = np.asarray(capOut)
        vids   = np.asarray(vids)

        logger.debug("Shape vids   %s [max distinct %d]" % (str(np.shape(vids)),len(idlst)))
        logger.debug("Shape afeatures  %s" % str(np.shape(afeatures)))
        logger.debug("Shape CapIn  %s" % str(np.shape(capIn)))
        logger.debug("Shape CapOut %s" % str(np.shape(capOut)))


        return [[capIn,afeatures,vids],capOut]
 
    def get_nextbatch(self, batch_size, arr_counter, ids):
        assert len(ids) > 0
        count = arr_counter[0]
        start = (count * batch_size) % len(ids)
        idlst = []
        for i in xrange(batch_size):
            idlst.append(ids[start])
            start = (start + 1) % len(ids)

        count = (count +1 ) % len(ids)
        arr_counter[0] = count
        return idlst
        
    '''
    typeSet 0:Training dataset, 1: Validation dataset, 2: Test Dataset
    '''
    # Sequential
    def data_generator(self, batch_size, start=0, typeSet = 0):
        if typeSet == 0:
            ids = self.vHandler.getTrainingIds()
        elif typeSet == 1:
            ids = self.vHandler.getValidationIds()
        elif typeSet == 2:
            ids = self.vHandler.getTestIds()
        else:
            assert False
        random.shuffle(ids)
        arr_counter = [0]
        count = (len(ids)+batch_size-1)//batch_size
        assert count > 0
        logger.debug("Max Batches of type %d : %d " % (typeSet, count))
        while True:
            idlst = self.get_nextbatch(batch_size, arr_counter, ids)
            data = self.datas_from_ids(idlst)
            ndata = []
            for d in data:
                if d is not None:
                    ndata.append(d)
            if len(ndata) > 0:
                yield ndata

    # Random
    def data_generator_random(self, batch_size, start=0, typeSet = 0):
        if typeSet == 0:
            ids = self.vHandler.getTrainingIds()
        elif typeSet == 1:
            ids = self.vHandler.getValidationIds()
        elif typeSet == 2:
            ids = self.vHandler.getTestIds()
        else:
            assert False
        random.shuffle(ids)
        count = (len(ids) + batch_size - 1)//batch_size
        assert count > 0
        if start == -1:
            start = random.randint(0,count)
        logger.debug("Max Batches of type %d : %d " % (typeSet, count))
        #start = start % count
        while True:
            bs = batch_size
            if bs>len(ids):
                bs=len(ids)
                logger.debug("FORCE Reducing Batch Size to %d from %d",bs,batch_size)
            idlst = random.sample(ids,bs)
            data = self.datas_from_ids(idlst)
            ndata = []
            for d in data:
                if d is not None:
                    ndata.append(d)
            if len(ndata) > 0:
                yield ndata
            #start = (start + 1)%count
