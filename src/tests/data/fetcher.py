import os

from common.config import get_tests_config


DIR_VIDEOS = get_tests_config()["dir_videos"]

def get_videopath(fname):
    '''Returns path of given test video file.'''
    return os.path.join(DIR_VIDEOS, fname)