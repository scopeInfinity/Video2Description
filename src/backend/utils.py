import re

def caption_tokenize(caption):
    caption = re.sub('[^a-zA-Z]+', ' ', caption.encode('utf-8')).lower()
    caption = caption.split()
    return caption
