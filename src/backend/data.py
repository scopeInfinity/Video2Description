import cv2
import json
import os
import re

from pytube import YouTube
from subprocess import check_output

DIR = 'Videos'
CATEGORIES = (1<<3)

with open('train_2017/videodatainfo_2017.json') as f:
    vdi = json.loads(f.read())
    _videos={}
    for v in vdi['videos']:
        if ((1<<v['category'])&CATEGORIES)>0:
            _videos[v['video_id']] = { 'url' : v['url'] }
    for s in vdi['sentences']:
        if s['video_id'] in _videos.keys():
            _videos[s['video_id']]['caption'] = s['caption']
    
def download_all():
    count = 0
    for _id in _videos.keys():
        print("Dowloading %s " % _id)
        getVideoFname(_id)
        count+=1
        print("%3.2f %% Completed" % (100.0*count/len(_videos.keys())))

def sz_videos():
    return len(_videos)

def get_videoId(index):
    v = _videos.keys()[index]
    return v

def getVideoFname(videoId):
    try:
        fname = DIR+"/"+videoId+".mp4"
        # Caching
        if os.path.isfile(fname):
            print("Used cached video file %s " % fname)
            return fname
        url = _videos[videoId]['url']
        print("Fetching info from %s " % url)
        yt = YouTube(url)
        v = yt.filter('mp4')[0]
        # For Non mp4, NOT SUPPORTED for now
        # if v is None:
        #     v = yt.videos()[0]
        dfname = DIR+"/"+v.filename+".mp4"
        if v:
            print("Video Downloading %s " % videoId)
            v.download(DIR)
            print("Moving %s to %s " % (dfname,fname))
            os.rename(dfname,fname)
            print("Video Downloaded")
            return fname
        else: 
            print("Video not Found for %s " % videoId)
            return None
    except Exception as e:
        print(str(e))
        return None
        

def getCaption(videoId):
    return _videos[videoId]['caption']

def getDuration(fname):
    return int(float(os.popen("ffprobe -i %s -show_format 2>&1 | grep duration | sed 's/duration=//'" % (fname,)).read()))

def getFrame(fname,ts):
    iname = 'Videos/frames/frame.png'
    hr = ts//3600
    ts = ts%(3600)
    mi = ts//60
    ts = ts%60
    time = "%02d:%02d:%02d" % (hr,mi,ts)
    print("getting frame for time %s " % time)
    os.popen("ffmpeg -y -ss %s -i %s -frames 1 %s" % (time,fname,iname))
    img = cv2.imread(iname)
    return img
    
def getVideo(videoId):
    fname = getVideoFname(videoId)
    if fname is None:
        return None
    print("Loading Video %s " % fname)
    duration = getDuration(fname)
    print("Duration " + str(duration) +" sec")
    COUNT = 5
    if duration < 15*COUNT:
        print("Video too short")
        return None
    frames = []
    for i in range(COUNT):
        image = getFrame(fname,15*(i+1))
        frames.append(image)
    return frames
    
 
