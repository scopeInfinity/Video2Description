import os

import config


DIR_VIDEOS = config.getTestsConfig()["dir_videos"]

def get_videopath(fname):
    '''Returns path of given test video file.'''
    return os.path.join(DIR_VIDEOS, fname)