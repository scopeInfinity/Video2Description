import os, urllib
import json
import urlparse
import time

class VideoHandler:
    fname_offset = "VideoDataset"
    fname_train = "videodatainfo_2017.json"
    SLEEPTIME = 30
    STHRES = 10*1024
    
    def __init__(self, maindir, fname):
        self.fname = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, fname)
        self.vdir = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, "videos")
        self.logfile = "%s/%s/%s" % (maindir, VideoHandler.fname_offset, "log.txt")
        with open(fname) as f:
            self.data = json.load(f)
        self.captions = dict()
        for sen in self.data['sentences']:
            self.captions[self.stringIdToInt(sen['video_id'])] = sen['caption']
            
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

    def getYoutubeId(self,url):
        query = urlparse.parse_qs(urlparse.urlparse(url).query)
        print query
        return query['v'][0]

    def downloadVideo(self,_id):
        video = self.data['videos'][_id]
        url = video['url']
        stime = video['start time']
        etime = video['end time']
        sfname = "%s/%d.mp4" % (self.vdir, _id)
        if os.path.exists(sfname):
            print "Video Id [%d] Already Downloaded" % _id
            return
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
            with open(self.logfile,"a") as f:
                f.write("Crosscheck file %d, %s with size %d\n" % (_id, youtubeId, fs))
            os.remove(sfname)
            open(sfname,'a').close()
        self.takebreak()

    def takebreak(self):
        time.sleep(VideoHandler.SLEEPTIME)


def autodownload():
    vHandler = VideoHandler("/home/gagan.cs14/btp/",VideoHandler.fname_train)
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

if __name__ == "__main__":
    autodownload()
