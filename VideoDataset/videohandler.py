import cv2
import os, urllib
import json
import urlparse
import time
import shutil
import argparse
import numpy as np
from pprint import pprint
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
class VideoHandler:
    LIMIT_FRAMES = 40
    SHAPE = (224, 224)
        
    fname_offset = "VideoDataset"
    fname_train = "videodatainfo_2017.json"
    SLEEPTIME = 120
    STHRES = 10*1024
    EXTRACT_COUNTER = 0 # For multiprocessing
    
    def __init__(self, maindir, fname):
        self.splitTrainValidTest = [90,5,5] # Out of 100
        self.fname = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, fname)
        self.vdir = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, "videos")
        self.cdir = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, "cache_"+str(self.LIMIT_FRAMES))
        self.tdir = "%s/%s" % (self.vdir, "extract")
        self.logfile = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, "log.txt")
        if os.path.exists(self.tdir):
            shutil.rmtree(self.tdir)
        os.mkdir(self.tdir)
        if not os.path.exists(self.cdir):
            os.mkdir(self.cdir)
        self.build_captions()

    def build_captions(self):
        with open(self.fname) as f:
            self.data = json.load(f)
        # id => [caption]
        self.captions = dict()
        idcreated = set()
        for sen in self.data['sentences']:
            _id = self.stringIdToInt(sen['video_id'])
            if _id not in idcreated:
                idcreated.add(_id)
                self.captions[_id] = []
            self.captions[_id].append(sen['caption'])

    def set_vmodel(self,vmodel):
        self.vmodel = vmodel

    def getCaptionData(self):
        return self.captions

    def stringIdToInt(self,sid):
        assert(sid[:5]=='video')
        return int(sid[5:])

    def getAllIds(self):
        return self.captions.keys()

    def getDownloadedIds(self):
        allfiles = os.listdir(self.vdir)
        vfiles = []
        for f in allfiles:
            if f.endswith(".mp4"):
                if os.path.getsize("%s/%s" % (self.vdir,f)) >= VideoHandler.STHRES:
                    vfiles.append(int(f[:-4]))
        return vfiles

    def filterMod100(self, lst, _min, _max):
        ids = []
        for i,_id in enumerate(lst):
            if (i%100)>=_min and (i%100)<_max:
                ids.append(_id)
        return ids
        
    def getTrainingIds(self):
        return self.filterMod100(self.getDownloadedIds(), 0, self.splitTrainValidTest[0])

    def getValidationIds(self):
        return self.filterMod100(self.getDownloadedIds(), self.splitTrainValidTest[0], self.splitTrainValidTest[0]+self.splitTrainValidTest[1])

    def getTestIds(self):
        return self.filterMod100(self.getDownloadedIds(), self.splitTrainValidTest[0]+self.splitTrainValidTest[1], 100)

    def getYoutubeId(self,url):
        query = urlparse.parse_qs(urlparse.urlparse(url).query)
        print query
        return query['v'][0]

    def downloadVideo(self, _id, logs = True):
        video = self.data['videos'][_id]
        url = video['url']
        stime = video['start time']
        etime = video['end time']
        sfname = "%s/%d.mp4" % (self.vdir, _id)
        if os.path.exists(sfname):
            if logs:
                print "Video Id [%d] Already Downloaded" % _id
            return sfname
        youtubeId = self.getYoutubeId(url)
        turl = "curl 'http://hesetube.com/download.php?id=%s'" % (youtubeId)
        durl = "http://hesetube.com/video/%s.mp4?start=%f&end=%f" % (youtubeId, stime, etime) 
        print durl
        os.system(turl)
        cont = urllib.urlopen(durl).read()
        with open(sfname,"wb") as f:
            f.write(cont)
            print "Video Id [%d] Downloaded : %s " % (_id, youtubeId)
        fs = os.path.getsize(sfname)
        if fs < VideoHandler.STHRES:
            print "Crosscheck failed, File Size : %d" % fs
            with open(self.logfile,"a") as f:
                f.write("Crosscheck file %d, %s with size %d\n" % (_id, youtubeId, fs))
            os.remove(sfname)
            open(sfname,'a').close()
            self.takebreak()
            return None
        else:
            self.takebreak()
            return sfname

    def takebreak(self):
        time.sleep(VideoHandler.SLEEPTIME)

    '''
    Either frames of video from id or vfname
    '''
    CRAZY = 0
    #@synchronized
    def get_crazy_id(self):
       VideoHandler.EXTRACT_COUNTER += 1
       return VideoHandler.EXTRACT_COUNTER
        
    def get_iframes_cached(self, _id):
        cfname = "%s/%d.npy" % (self.cdir, _id)
        if os.path.exists(cfname):
            f = open(cfname, 'rb')
            frames = np.load(f)
            assert len(frames) == self.LIMIT_FRAMES
            return frames
        return None

    def cached_iframe(self, _id, frames):
        cfname = "%s/%d.npy" % (self.cdir, _id)
        print "Cached %s" % cfname
        with open(cfname, 'wb') as f:
            np.save(f,frames)

    def get_iframes(self, _id = None, sfname = None, logs = True):
        assert (_id is None) ^ (sfname is None)
        # Load if cached
        if _id is not None:
            rframes = self.get_iframes_cached(_id)
            if rframes is not None:
                return rframes

        # Load frames from file
        if sfname is None:
            sfname = self.downloadVideo(_id, logs)
        if sfname is None:
            return None
        vcap = cv2.VideoCapture(sfname)
        success, frame = vcap.read()
        allframes = []
        while True:
            success, frame = vcap.read()
            if not success:
                break
            allframes.append(cv2.resize(frame, self.SHAPE))
        if len(allframes) < self.LIMIT_FRAMES:
            print "File [%s] with limited frames (%d)" % (sfname, len(allframes))
            return None

        period = len(allframes) / self.LIMIT_FRAMES
        rframes = allframes[:period * self.LIMIT_FRAMES:period]
        frames_out = self.vmodel.preprocess_partialmodel(rframes)

        # Cache it
        if _id is not None:
            self.cached_iframe(_id, frames_out)
        return frames_out

    def get_frames(self,_id = None, sfname = None, logs = True):
        assert (_id is None) ^ (sfname is None)
        if sfname is None:
            sfname = self.downloadVideo(_id, logs)
        if sfname is None:
            return None
        edir = "%s/v_%d" % (self.tdir, self.get_crazy_id())
        if os.path.exists(edir):
            shutil.rmtree(edir)
        os.mkdir(edir)
        cmd = "ffmpeg -i %s -vf fps=%d -s 224x224 %s/0_%%03d.jpg &> /dev/null" % (sfname, 5, edir) #&> /dev/null
        if logs:
            print cmd
        returnStatus = os.system(cmd)
        if returnStatus != 0:
            print "Extracting Failed : %s" % sfname
            if os.path.exists(edir):
                print cmd
                print "Dir Exists"
                #shutil.rmtree(edir)
            return None
        files = os.listdir(edir)
        files = [("%s/%s"%(edir,f)) for f in files]
        LIMIT_FRAMES = 10
        if len(files)<LIMIT_FRAMES:
            return None
        # TODO : Pick frames uniformly
        files = files[:LIMIT_FRAMES]
        return (edir, files)

    def preprocess_frame(self, img):
        x = image.img_to_array(img)
        x /= 255.
        x -= 0.5
        x *= 2.
        return np.asarray(x)

    def assign_partial_model(self, partial_model):
        self.partial_model = partial_model

    #@synchronized
    def free_frames(self, edir):
        if edir is not None and os.path.exists(edir):
            try:
                shutil.rmtree(edir)
            except Exception as e:
                print(str(e))

def autodownload():
    print "Current Downloaded files"
    print vHandler.getDownloadedIds()
    vHandler.takebreak()
    print "Downloading More!!!"
    allIds = vHandler.getAllIds()
    tot =  len(allIds)
    for i,_id in enumerate(allIds):
        vHandler.downloadVideo(_id)
        percent = 100.0*(i+1)/tot
        print "%.3f Completed!" % percent

def show_counts():
    print "Training Videos   : %d " % len(vHandler.getTrainingIds())
    print "Validation Videos : %d " % len(vHandler.getValidationIds())
    print "Test Videos       : %d " % len(vHandler.getTestIds())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sc", "--show-count", help="show count for training/validation/test videos", action='store_true')
    parser.add_argument("-d", "--download", help="download more videos to extend dataset", action='store_true')
    parser.add_argument("-strain", "--show-train", help="show ids for training videos", action='store_true')
    parser.add_argument("-stest", "--show-test", help="show ids for test videos", action='store_true')
    parser.add_argument("-sval", "--show-val", help="show ids for validation videos", action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    vHandler = VideoHandler("/home/gagan.cs14/btp/",VideoHandler.fname_train)
    if args.show_count:
        show_counts()
    if args.download:
        autodownload()
    if args.show_train:
        print "Train Ids"
        pprint(vHandler.getTrainingIds())
    if args.show_test:
        print "Test Ids"
        pprint(vHandler.getTestIds())
    if args.show_val:
        print "Validation Ids"
        pprint(vHandler.getValidationIds())


