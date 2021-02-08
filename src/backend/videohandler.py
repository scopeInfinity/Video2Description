import argparse
import cv2
import time
import json
import numpy as np
import shutil
import os
import six.moves.urllib as urllib
import six.moves.urllib.parse as urllibparse
import librosa

from pprint import pprint

from common.config import get_app_config

DIR_VIDEO_DATASET = get_app_config()["VIDEOS_DATASET"]


class VideoHandler:
    LIMIT_FRAMES = 40
    AUDIO_FEATURE = (80, 40) #  TimeSamples, n_mfcc

    # ResNet
    SHAPE = (224, 224)
    ## InceptionV3
    # SHAPE = (299, 299)

    s_fname_train = "videodatainfo_2017.json"
    s_fname_test = "test_videodatainfo_2017.json"
    SLEEPTIME = 30
    STHRES = 10*1024
    EXTRACT_COUNTER = 0 # For multiprocessing
    
    def __init__(self, s_fname_train, s_fname_test):
        self.splitTrainValid = [95,5] # Out of 100
        self.fname_train = os.path.join(DIR_VIDEO_DATASET, s_fname_train)
        self.fname_test = os.path.join(DIR_VIDEO_DATASET, s_fname_test)
        self.vdir = os.path.join(DIR_VIDEO_DATASET, "videos")
        self.cdir = os.path.join(DIR_VIDEO_DATASET, "cache_"+str(self.LIMIT_FRAMES)+"_"+("%dx%d" % VideoHandler.SHAPE))
        self.adir = os.path.join(DIR_VIDEO_DATASET, "cache_audio_"+("%dx%d" % VideoHandler.AUDIO_FEATURE))
        self.tdir = os.path.join(self.vdir, "extract")
        self.logfile = os.path.join(DIR_VIDEO_DATASET, "log.txt")
        if not os.path.exists(self.vdir):
            os.mkdir(self.vdir)
        if os.path.exists(self.tdir):
            shutil.rmtree(self.tdir)
        os.mkdir(self.tdir)
        if not os.path.exists(self.cdir):
            os.mkdir(self.cdir)
        if not os.path.exists(self.adir):
            os.mkdir(self.adir)
        self.build_captions()

    def build_captions(self):
        with open(self.fname_train) as f:
            data_train = json.load(f)
        with open(self.fname_test) as f:
            data_test = json.load(f)
        self.vdata = dict()
        for item in data_train['videos']:
            self.vdata[item['id']] = item
        for item in data_test['videos']:
            self.vdata[item['id']] = item

        # id => [caption]
        self.captions = dict()
        idcreated = set()
        # Training Set
        for sen in data_train['sentences']:
            _id = self.stringIdToInt(sen['video_id'])
            if _id not in idcreated:
                idcreated.add(_id)
                self.captions[_id] = []
            self.captions[_id].append(sen['caption'])
        self.train_ids = list(idcreated)

        idcreated = set()
        # Test Set
        for sen in data_test['sentences']:
            _id = self.stringIdToInt(sen['video_id'])
            if _id not in idcreated:
                idcreated.add(_id)
                self.captions[_id] = []
            self.captions[_id].append(sen['caption'])
        self.test_ids = list(idcreated)

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
            # Issue: getDownloadedIds is called before creation of *_ignore file
            #       Program crases onces for creating these files then works normally
            if f.endswith(".mp4") and (not os.path.exists(os.path.join(self.vdir,f+"_ignore"))):
                if os.path.getsize("%s/%s" % (self.vdir,f)) >= VideoHandler.STHRES:
                    vfiles.append(int(f[:-4]))
        return vfiles

    def filterMod100(self, parentlst, lst, _min, _max):
        parentlst = set(parentlst)
        lst = set(lst)
        flst = lst.intersection(parentlst)
        lst = list(flst)
        ids = []
        for i,_id in enumerate(lst):
            if (i%100)>=_min and (i%100)<_max:
                ids.append(_id)
        return ids
        
    def getTrainingIds(self):
        return self.filterMod100(self.get_otrain_ids(), self.getDownloadedIds(), 0, self.splitTrainValid[0])

    def getValidationIds(self):
        return self.filterMod100(self.get_otrain_ids(), self.getDownloadedIds(), self.splitTrainValid[0],100)

    def getTestIds(self):
        return self.filterMod100(self.get_otest_ids(), self.getDownloadedIds(), 0, 100)

    def get_otrain_ids(self):
        return self.train_ids

    def get_otest_ids(self):
        return self.test_ids

    def getYoutubeId(self,url):
        query = urllibparse.parse_qs(urllibparse.urlparse(url).query)
        print(query)
        return query['v'][0]

    def downloadVideo(self, _id, logs = True):
        video = self.vdata[_id]
        url = video['url']
        stime = video['start time']
        etime = video['end time']
        sfname = "%s/%d.mp4" % (self.vdir, _id)
        if os.path.exists(sfname):
            if logs:
                print("Video Id [%d] Already Downloaded" % _id)
            return sfname
        youtubeId = self.getYoutubeId(url)
        turl = "curl 'https://hesetube.com/download.php?id=%s'" % (youtubeId)
        durl = "https://hesetube.com/video/%s.mp4?start=%f&end=%f" % (youtubeId, stime, etime) 
        print(durl)
        print(turl)
        os.system(turl)
        cont = urllib.urlopen(durl).read()
        with open(sfname,"wb") as f:
            f.write(cont)
            print("Video Id [%d] Downloaded : %s " % (_id, youtubeId))
        fs = os.path.getsize(sfname)
        if fs < VideoHandler.STHRES:
            print("Crosscheck failed, File Size : %d" % fs)
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

    def get_audio_cached(self, _id):
        afname = "%s/%d.npy" % (self.adir, _id)
        if os.path.exists(afname):
            f = open(afname, 'rb')
            feature = np.load(f)
            if np.shape(feature) != self.AUDIO_FEATURE:
                print("Feature Shape error at %d, %s" % (_id, np.shape(feature)))
            assert np.shape(feature) == self.AUDIO_FEATURE
            return feature
        return None

    def cached_iframe(self, _id, frames):
        cfname = "%s/%d.npy" % (self.cdir, _id)
        print("Cached %s" % cfname)
        with open(cfname, 'wb') as f:
            np.save(f,frames)

    def cached_audio(self, _id, feature):
        afname = "%s/%d.npy" % (self.adir, _id)
        print("Cached %s" % afname)
        with open(afname, 'wb') as f:
            np.save(f,feature)

    def file_to_videofeature(self, sfname):
        vcap = cv2.VideoCapture(sfname)
        success, frame = vcap.read()
        allframes = []
        while True:
            success, frame = vcap.read()
            if not success:
                break
            allframes.append(cv2.resize(frame, VideoHandler.SHAPE))
        if len(allframes) < self.LIMIT_FRAMES:
            print("File [%s] with limited frames (%d)" % (sfname, len(allframes)))
            # Ignore those videos
            os.system("touch %s_ignore" % sfname)
            return None

        period = len(allframes) // self.LIMIT_FRAMES
        rframes = allframes[:period * self.LIMIT_FRAMES:period]
        frames_out = self.vmodel.preprocess_partialmodel(rframes)
        return frames_out

    def file_to_audiofeature(self, sfname):
        audio_y, sr = librosa.load(sfname)
        afeatures = librosa.feature.mfcc(y=audio_y, sr=sr, n_mfcc=self.AUDIO_FEATURE[1])
        afeatures = np.transpose(afeatures)
        ll = len(afeatures)
        parts = ll//self.AUDIO_FEATURE[0]
        division = []
        for i in range(self.AUDIO_FEATURE[0] - 1):
            division.append((i+1)*parts)
        for i in range(ll%self.AUDIO_FEATURE[0]):#left over
            division[i]+=1           
        afeatures = np.split(np.array(afeatures), division)
        afeature_out = []
        for af in afeatures:
            afeature_out.append(np.mean(np.array(af),axis = 0))
        afeature_out = np.asarray(afeature_out)
        if np.shape(afeature_out) != self.AUDIO_FEATURE:
            print("File [%s] with audio problem (%s)" % (sfname, str(np.shape(afeature_out))))
            # Ignore videos
            os.system("touch %s_ignore" % sfname)
        return afeature_out

    # (Video Feature, Audio Feature)
    def get_iframes_audio(self, _id = None, sfname = None, logs = True, cache_id = None):
        assert (_id is None) ^ (sfname is None)
        # Load if cached
        frames_out = None
        afeature_out = None
        if _id is not None or cache_id is not None:
            if _id is not None:
                cache_id = _id
            frames_out = self.get_iframes_cached(cache_id)
            afeature_out = self.get_audio_cached(cache_id)
            if frames_out is not None and afeature_out is not None:
                return (frames_out, afeature_out)
        # Load frames from file
        if sfname is None:
            sfname = self.downloadVideo(_id, logs)
        if sfname is None:
            return None

        to_cache_video = False
        to_cache_audio = False

        if frames_out is None:
            frames_out = self.file_to_videofeature(sfname)
            to_cache_video = True

        if afeature_out is None:
            afeature_out = self.file_to_audiofeature(sfname)
            to_cache_audio = True

        # Cache it
        if _id is not None or cache_id is not None:
            if _id is not None:
                cache_id = _id
            if to_cache_video:
                self.cached_iframe(cache_id, frames_out)
            if to_cache_audio:
                self.cached_audio(cache_id, afeature_out)
        return (frames_out, afeature_out)

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
        cmd = "ffmpeg -i %s -vf fps=%d -s %dx%d %s/0_%%03d.jpg &> /dev/null" % (
                   sfname, 5, VideoHandler.SHAPE[0], VideoHandler.SHAPE[1], edir) #&> /dev/null
        if logs:
            print(cmd)
        returnStatus = os.system(cmd)
        if returnStatus != 0:
            print("Extracting Failed : %s" % sfname)
            if os.path.exists(edir):
                print(cmd)
                print("Dir Exists")
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
    print("Current Downloaded files")
    print(vHandler.getDownloadedIds())
    #vHandler.takebreak()
    print("Downloading More!!!")
    allIds = vHandler.getAllIds()
    tot =  len(allIds)
    for i,_id in enumerate(allIds):
        vHandler.downloadVideo(_id)
        percent = 100.0*(i+1)/tot
        print("%.3f Completed!" % percent)

def cache_videoid(_id, percent):
    print("%.3f caching scheduled!" % percent)
    vHandler.get_iframes_audio(_id = _id)

def autocache():
    import sys
    import concurrent.futures
    sys.path.append("..")
    
    from model import VModel
    vmodel = VModel(-1,-1, cutoffonly = True)
    vHandler.set_vmodel(vmodel)
    ids = vHandler.getDownloadedIds()
    tot = len(ids)

    # from concurrent.futures import ThreadPoolExecutor, wait
    # pool = ThreadPoolExecutor(10)
    futures = []
    for i,_id in enumerate(ids):
        percent = 100.0*(i+1)/tot
        #futures.append(pool.submit(cache_videoid, _id, percent))
        cache_videoid(_id, percent)
    #print(wait(futures))
    print("Caching Completed!")

def show_counts():
    print("Training Videos   : %d " % len(vHandler.getTrainingIds()))
    print("Validation Videos : %d " % len(vHandler.getValidationIds()))
    print("Test Videos       : %d " % len(vHandler.getTestIds()))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sc", "--show-count", help="show count for training/validation/test videos", action='store_true')
    parser.add_argument("-d", "--download", help="download more videos to extend dataset", action='store_true')
    parser.add_argument("-ac", "--auto-cache", help="cache all downloaded videos", action='store_true')
    parser.add_argument("-strain", "--show-train", help="show ids for training videos", action='store_true')
    parser.add_argument("-stest", "--show-test", help="show ids for test videos", action='store_true')
    parser.add_argument("-sval", "--show-val", help="show ids for validation videos", action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    vHandler = VideoHandler(VideoHandler.s_fname_train, VideoHandler.s_fname_test)
    if args.show_count:
        show_counts()
        exit()
    if args.download:
        autodownload()
        exit()
    if args.auto_cache:
        autocache()
        exit()
    if args.show_train:
        print("Train Ids")
        pprint(vHandler.getTrainingIds())
    if args.show_test:
        print("Test Ids")
        pprint(vHandler.getTestIds())
    if args.show_val:
        print("Validation Ids")
        pprint(vHandler.getValidationIds())


