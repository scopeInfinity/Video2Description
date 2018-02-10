import cv2
import os, urllib
import json
import urlparse
import time
import shutil
import argparse

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
        self.tdir = "%s/%s" % (self.vdir, "extract")
        self.logfile = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, "log.txt")
        if os.path.exists(self.tdir):
            shutil.rmtree(self.tdir)
        os.mkdir(self.tdir)
        with open(self.fname) as f:
            self.data = json.load(f)
        self.captions = dict()
        for sen in self.data['sentences']:
            self.captions[self.stringIdToInt(sen['video_id'])] = sen['caption']

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
        
    def get_iframes(self, _id = None, sfname = None, logs = True):
        assert (_id is None) ^ (sfname is None)
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
            return None
        period = len(allframes) / self.LIMIT_FRAMES
        rframes = allframes[:period * self.LIMIT_FRAMES:period]
        return rframes

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
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    vHandler = VideoHandler("/home/gagan.cs14/btp/",VideoHandler.fname_train)
    if args.show_count:
        show_counts()
    if args.download:
        autodownload()

